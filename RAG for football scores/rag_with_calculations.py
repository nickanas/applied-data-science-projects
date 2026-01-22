import json
import re
from datetime import datetime
from langchain_core.documents import Document
import pandas as pd
import numpy as np
from langchain_huggingface.llms import HuggingFacePipeline
from langchain_huggingface.embeddings import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableParallel, RunnablePassthrough, RunnableLambda
from sklearn.metrics import accuracy_score, f1_score
from langchain_community.retrievers import BM25Retriever
import re

data = json.load(open("en.1.json"))
season = data.get("name", "Premier League 2024/25")
matches = data["matches"]

def to_int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default

structured_matches = []

for m in matches:
    ht = (m.get("score", {}) or {}).get("ht")
    ft = (m.get("score", {}) or {}).get("ft")

    ht_home = to_int(ht[0], None) if isinstance(ht, (list, tuple)) and len(ht) >= 2 else None
    ht_away = to_int(ht[1], None) if isinstance(ht, (list, tuple)) and len(ht) >= 2 else None

    ft_home = to_int(ft[0], None) if isinstance(ft, (list, tuple)) and len(ft) >= 2 else None
    ft_away = to_int(ft[1], None) if isinstance(ft, (list, tuple)) and len(ft) >= 2 else None

    structured_matches.append({
        "matchday": m["round"],
        "date": m["date"],
        "time": m["time"],
        "home_team": m["team1"],
        "away_team": m["team2"],
        "ht_home": ht_home, "ht_away": ht_away,
        "ft_home": ft_home, "ft_away": ft_away,
    })

team_matches = {}

for t in structured_matches:
    matchday_integers = [int(s) for s in t["matchday"].split() if s.isdigit()]
    matchday_integer = matchday_integers[0]
         
    if t["home_team"] not in team_matches:
        team_matches[t["home_team"]] = []
    
    team_matches[t["home_team"]].append({
        "matchday number": matchday_integer, 
        "round": t["matchday"], 
        "date": t["date"], 
        "time": t["time"],
        "team": t["home_team"], 
        "opponent": t["away_team"], 
        "venue": "home",
        "ht_for": t["ht_home"], 
        "ht_against": t["ht_away"],
        "ft_for": t["ft_home"], 
        "ft_against": t["ft_away"]
    })
        
    if t["away_team"] not in team_matches:
        team_matches[t["away_team"]] = []
    
    team_matches[t["away_team"]].append({
        "matchday number": matchday_integer, 
        "round": t["matchday"], 
        "date": t["date"], 
        "time": t["time"],
        "team": t["away_team"], 
        "opponent": t["home_team"], 
        "venue": "away",
        "ht_for": t["ht_away"], 
        "ht_against": t["ht_home"],
        "ft_for": t["ft_away"], 
        "ft_against": t["ft_home"]
    })

for team in team_matches:
    team_matches[team].sort(key=lambda x: x["matchday number"])
    team_matches[team].sort(key=lambda x: x["time"])
    team_matches[team].sort(key=lambda x: x["date"])

blocks = [(1,8), (9,16), (17,24), (25,32), (33,38)]

def build_documents(summary: bool):
    docs = []

    for team in team_matches:
        matches = team_matches[team]

        for block in blocks:
            first_md, last_md = block

            matches_block = []

            for m in matches:
                if first_md <= m["matchday number"] <= last_md:
                    matches_block.append(m)

            points = 0
            wins = 0
            draws = 0
            losses = 0
            goals_for = 0
            goals_against = 0

            for m in matches_block:
                goals_for += m["ft_for"]
                goals_against += m["ft_against"]

                if m["ft_for"] > m["ft_against"]:
                    wins += 1
                    points += 3
                elif m["ft_for"] == m["ft_against"]:
                    draws += 1
                    points += 1
                else:
                    losses += 1

            document_text = f"""
TEAM: {team}
MATCHDAYS: {first_md}-{last_md}\n\n
League: Premier League
SEASON: {season}"""

            for m in matches_block:
                document_text += (
                    f"{m['round']}\n"
                    f"{m['date']} {m['time']}\n"
                    f"{m['team']} ({m['venue']}) vs {m['opponent']}\n"
                    f"Half time score: {m['ht_for']}-{m['ht_against']}\n"
                    f"Full time score: {m['ft_for']}-{m['ft_against']}\n\n"
                )

            if summary:
                document_text += (
                    "Summary for matchdays:\n"
                    f"Games: {len(matches_block)}\n"
                    f"Points: {points}\n"
                    f"Wins: {wins}\n"
                    f"Draws: {draws}\n"
                    f"Losses: {losses}\n"
                    f"Goals for: {goals_for}\n" 
                    f"Goals against: {goals_against}\n" 
                    f"Goals difference: {goals_for-goals_against}"
                )

            doc_id = f"{team}__md{first_md}-{last_md}"

            docs.append(Document(
                page_content=document_text,
                metadata={"doc_id": doc_id, "team": team, "md_start": first_md, "md_end": last_md}
            ))
          
    return docs

docs_summary = build_documents(summary=True)
docs_no_summary = build_documents(summary=False)

##############################################################
# Python Calculations
##############################################################

def extract_matches_for_team(team_name, start_md, end_md):
    """Extract matches for a specific team and matchday range"""
    if team_name not in team_matches:
        return []
    
    matches = team_matches[team_name]
    filtered = [
        m for m in matches 
        if start_md <= m['matchday number'] <= end_md
    ]
    return filtered

def calculate_team_stats(team_name, ranges):
    matches = team_matches.get(team_name, [])
    chosen = [m for m in matches if any(s <= m["matchday number"] <= e for s, e in ranges)]

    if not chosen:
        return None

    points = wins = draws = losses = 0
    goals_for = goals_against = 0

    for m in chosen:
        gf = m["ft_for"]
        ga = m["ft_against"]
        if gf is None or ga is None:
            return None

        goals_for += gf
        goals_against += ga

        if gf > ga:
            wins += 1
            points += 3
        elif gf == ga:
            draws += 1
            points += 1
        else:
            losses += 1

    return {
        "points": points,
        "goals_for": goals_for,
        "goals_against": goals_against,
        "goal_difference": goals_for - goals_against,
        "wins": wins,
        "draws": draws,
        "losses": losses,
        "matches_played": len(chosen),
    }

def find_team_in_question(question_lower: str, team_names):
    q = re.sub(r'[^a-z0-9\s]', ' ', question_lower)
    q = re.sub(r'\s+', ' ', q).strip()

    candidates = []
    for team in team_names:
        t = team.lower()

        aliases = set()
        aliases.add(t)
        aliases.add(t.replace(" fc", "").strip())
        aliases.add(t.replace(" afc ", " ").strip())

        if "manchester city" in t:
            aliases.add("man city")
            aliases.add("manchester city")
        if "manchester united" in t:
            aliases.add("man united")
            aliases.add("manchester united")

        for a in aliases:
            a_norm = re.sub(r'[^a-z0-9\s]', ' ', a)
            a_norm = re.sub(r'\s+', ' ', a_norm).strip()

            if re.search(rf"\b{re.escape(a_norm)}\b", q):
                candidates.append((len(a_norm), team))

    if not candidates:
        return None

    candidates.sort(reverse=True)
    return candidates[0][1]

def parse_matchday_ranges(question: str):
    q = question.lower()
    ranges = []

    for a, b in re.findall(r"matchdays?\s+(\d+)\s*(?:-|to)\s*(\d+)", q):
        ranges.append((int(a), int(b)))

    for a, b in re.findall(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b", q):
        a, b = int(a), int(b)
        if 1 <= a <= 38 and 1 <= b <= 38:
            ranges.append((a, b))

    for a in re.findall(r"\bmatchday\s+(\d+)\b", q):
        md = int(a)
        ranges.append((md, md))

    if "whole season" in q or "entire season" in q or "in the season" in q:
        return [(1, 38)]

    if not ranges:
        return [(1, 38)]

    ranges.sort()
    merged = []
    for s, e in ranges:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]

def parse_question_for_calculation(question):
    question_lower = question.lower()

    blocked_phrases = [
        "against", "vs", "versus",
        "home game", "away game", "both games",
        "in the games", "in both games",
        "than", "more than", "difference between", "how many more"
    ]

    if any(p in question_lower for p in blocked_phrases):
        return {"team": find_team_in_question(question_lower, team_matches.keys()), "start_md": 1, "end_md": 38, "stat_type": None}

    team_found = find_team_in_question(question_lower, team_matches.keys())

    # use parse_matchday_ranges instead of manual regex because it handles multiple ranges correctly
    ranges = parse_matchday_ranges(question)

    # For backwards compatibility, set start_md and end_md from the ranges
    if ranges:
        start_md = ranges[0][0]  # Start of first range
        end_md = ranges[-1][1]   # End of last range
    else:
        start_md = 1
        end_md = 38

    if any(phrase in question_lower for phrase in ['first half', 'half time', 'halftime', 'half-time', ' ht ', 'half ']):
        return {
            "team": team_found,
            "start_md": start_md,
            "end_md": end_md,
            "stat_type": None
        }

    stat_type = None

    if "goals against" in question_lower or "conced" in question_lower:
        stat_type = "goals_against"
    elif "goal difference" in question_lower:
        stat_type = "goal_difference"
    elif "goals" in question_lower and ("score" in question_lower or "scor" in question_lower or "in total" in question_lower):
        stat_type = "goals_for"
    elif "points" in question_lower:
        stat_type = "points"
    elif "wins" in question_lower or "won" in question_lower:
        stat_type = "wins"
    elif "draws" in question_lower or "drawn" in question_lower:
        stat_type = "draws"
    elif "losses" in question_lower or "lost" in question_lower:
        stat_type = "losses"

    return {
        "team": team_found,
        "start_md": start_md,
        "end_md": end_md,
        "stat_type": stat_type
    }

##############################################################
# Hybrid Python + LLM Function
##############################################################

def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)

def get_python_calculation_context(question):
    """Get Python calculation as context string (if possible)"""
    parsed = parse_question_for_calculation(question)
    
    if not (parsed["team"] and parsed["stat_type"]):
        return ""
    
    ranges = parse_matchday_ranges(question)
    stats = calculate_team_stats(parsed["team"], ranges)
    
    if not stats:
        return ""
    
    result = stats[parsed["stat_type"]]
    
    python_context = f"""
=== PYTHON CALCULATION RESULT ===
Team: {parsed["team"]}
Matchday ranges: {ranges}
Matches analyzed: {stats['matches_played']}

Calculated {parsed["stat_type"]}: {result}

Full stats for verification:
- Points: {stats['points']}
- Wins: {stats['wins']}
- Draws: {stats['draws']}
- Losses: {stats['losses']}
- Goals for: {stats['goals_for']}
- Goals against: {stats['goals_against']}
- Goal difference: {stats['goal_difference']}

NOTE: You should verify this calculation matches the context documents below and use it in your answer.
=== END PYTHON CALCULATION ===
"""
    return python_context

def answer_with_hybrid_approach(question, retrieved_docs, chain):
    """Always get LLM answer, but provide Python calculation as additional context"""

    # Get Python calculation context (empty string if not applicable)
    python_context = get_python_calculation_context(question)

    # FORMAT retrieved documents
    docs_context = format_docs(retrieved_docs)

    # Combine Python context with document context
    if python_context:
        enhanced_context = python_context + "\n\n" + docs_context
    else:
        enhanced_context = docs_context

    # ADD THIS DEBUG LOGGING:
    print(f"\n{'='*60}")
    print(f"DEBUG - Question: {question[:80]}")
    print(f"DEBUG - Python context present: {bool(python_context)}")
    if python_context:
        # Print just the answer line from Python context
        import re
        match = re.search(r'Calculated \w+: (\d+)', python_context)
        if match:
            print(f"DEBUG - Python computed: {match.group(1)}")
    print(f"DEBUG - Context length: {len(enhanced_context)} chars")
    # END DEBUG LOGGING

    # Call LLM with enhanced context
    llm_result = chain.invoke({
        "context": enhanced_context,
        "question": question
    })

    # debug
    print(f"DEBUG - LLM output: {llm_result[:100]}...")
    print(f"{'='*60}\n")

    return {
        "answer": llm_result,
        "had_python_calc": bool(python_context),
        "python_context": python_context
    }

##############################################################
# Load Questions
##############################################################

with open("questions100.json") as f:
    questions = json.load(f)

doc_by_id = {doc.metadata["doc_id"]: doc for doc in docs_no_summary}

def print_documents_for_question(question, doc_by_id):
    print("=" * 80)
    print(f"QID: {question['qid']}")
    print(f"Question: {question['question']}")
    print(f"Difficulty: {question['difficulty']}")
    print(f"Gold answer: {question['answer']}")
    print("=" * 80)

    for doc_id in question["gold_document_ids"]:
        doc = doc_by_id[doc_id]

        print("\n" + "-" * 60)
        print(f"Document ID: {doc.metadata['doc_id']}")
        print(f"Team: {doc.metadata['team']}")
        print(f"Matchdays: {doc.metadata['md_start']}-{doc.metadata['md_end']}")
        print("-" * 60)

        print(doc.page_content)

##############################################################
# Initialize Models
##############################################################

model = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mathstral-7B-v0.1",
    task="text-generation",
    pipeline_kwargs={"return_full_text": False, "do_sample": False},
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")

def extract_label(answer: str):
    text = re.search(r"FINAL:\s*(.*)", answer)
    if not text:
        return None
    label = text.group(1).strip()
    return label

def gold_value(q):
    # Handle simple_test_questions.json format (has 'gold_answer')
    if "gold_answer" in q:
        return int(q["gold_answer"])

    # Handle questions100.json format
    if q.get("answer_type") == "number":
        return int(q["answer"]["value"])
    return q["answer"]["value"].strip()

##############################################################
# HYBRID APPROACH: No Summary + Python as Context
##############################################################

vector_store_no_summary = Chroma.from_documents(
    documents=docs_no_summary, 
    embedding=embeddings, 
    collection_name="pl_no_summary_hybrid"
)

retriever_no_summary = vector_store_no_summary.as_retriever(search_kwargs={"k": 5})

template_hybrid_no_summary = """
!!! CRITICAL INSTRUCTION !!!
If you see "=== PYTHON CALCULATION RESULT ===" below, you MUST do the following:
1. Look for the line that says "Calculated [stat_type]: [number]"
2. Use that EXACT number in your FINAL answer
3. DO NOT count matches manually
4. DO NOT add numbers from different matchday blocks yourself
5. The Python calculation is always correct - trust it completely

You are answering questions about the Premier League 2024/25 season.

Context documents show ONE team and ONE matchday block each:
Blocks: 1-8, 9-16, 17-24, 25-32, 33-38

Rules:
1) If Python calculation provided: USE IT. Do not override.
2) If no Python: Use documents to answer
3) Output EXACTLY two lines

Output format:
FINAL: <number>
EVIDENCE: <mention Python calculation if provided>

Context:
{context}

Question:
{question}
"""

prompt_hybrid_no_summary = ChatPromptTemplate.from_template(template_hybrid_no_summary)

runnable_parallel_hybrid_no_summary = RunnableParallel(
    context=retriever_no_summary, 
    question=RunnablePassthrough()
)

chain_hybrid_no_summary = (
    {"context": RunnableLambda(lambda x: x["context"]),
     "question": RunnableLambda(lambda x: x["question"])}
    | prompt_hybrid_no_summary
    | model
    | StrOutputParser()
)

stats_hybrid_no_summary = {
    "all": {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    "number": {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0}
}

stats_difficulty_hybrid_no_summary = {
    1: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    2: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    3: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    4: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    5: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0}
}

failed_questions_hybrid_no_summary = []

print("\n" + "="*80)
print("Starting evaluation: HYBRID NO SUMMARY (Python + LLM)")
print("="*80)

for i, q in enumerate(questions, 1):
    print(f"Processing question {i}/{len(questions)}...", end='\r')
    
    retrieval_result = runnable_parallel_hybrid_no_summary.invoke(q["question"])
    retrieved_docs = retrieval_result["context"]
    
    # Always use hybrid approach
    result = answer_with_hybrid_approach(
        q["question"], 
        retrieved_docs, 
        chain_hybrid_no_summary
    )
    
    final_answer = result['answer']
    had_python = result['had_python_calc']
    
    predicted_label = extract_label(final_answer)
    gold_label = gold_value(q)

    # Convert to int if needed
    if predicted_label and predicted_label.isdigit():
        predicted_label = int(predicted_label)

    retrieved_ids = [d.metadata["doc_id"] for d in retrieved_docs]

    # Handle both formats (simple_test_questions.json doesn't have gold_document_ids)
    if "gold_document_ids" in q:
        gold_ids = q["gold_document_ids"]
        retrieval_hits = sum(1 for g in gold_ids if g in retrieved_ids)
        recall = retrieval_hits / len(gold_ids)
    else:
        recall = 1.0  # Default for simple test questions without gold doc IDs
    
    correct = (predicted_label == gold_label)
    
    if not correct:
        failed_questions_hybrid_no_summary.append({
            'question': q['question'],
            'difficulty': int(q['difficulty']),
            'predicted': predicted_label,
            'gold': gold_label,
            'had_python': had_python,
            'answer': final_answer
        })

    stats_hybrid_no_summary["all"]["total"] += 1
    stats_hybrid_no_summary["all"]["recall"] += recall
    if had_python:
        stats_hybrid_no_summary["all"]["had_python"] += 1
    if correct:
        stats_hybrid_no_summary["all"]["correct"] += 1
    
    answer_type = q["answer_type"]
    stats_hybrid_no_summary[answer_type]["total"] += 1
    stats_hybrid_no_summary[answer_type]["recall"] += recall
    if had_python:
        stats_hybrid_no_summary[answer_type]["had_python"] += 1
    if correct:
        stats_hybrid_no_summary[answer_type]["correct"] += 1
    
    difficulty = int(q["difficulty"])
    stats_difficulty_hybrid_no_summary[difficulty]["total"] += 1
    stats_difficulty_hybrid_no_summary[difficulty]["recall"] += recall
    if had_python:
        stats_difficulty_hybrid_no_summary[difficulty]["had_python"] += 1
    if correct:
        stats_difficulty_hybrid_no_summary[difficulty]["correct"] += 1

print("\n" + "="*80)
print("Failed Questions Analysis - HYBRID NO SUMMARY")
print("="*80)
print(f"Total failed: {len(failed_questions_hybrid_no_summary)} out of {len(questions)}")

for difficulty in [1, 2, 3, 4, 5]:
    failed_at_diff = [f for f in failed_questions_hybrid_no_summary if f['difficulty'] == difficulty]
    print(f"\n--- Difficulty {difficulty}: {len(failed_at_diff)} failed ---")
    
    for i, f in enumerate(failed_at_diff[:3], 1):
        print(f"\n{i}. {f['question']}")
        print(f"   Predicted: {f['predicted']} | Gold: {f['gold']} | Had Python: {f['had_python']}")
        if len(f['answer']) > 150:
            print(f"   Answer: {f['answer'][:150]}...")
        else:
            print(f"   Answer: {f['answer']}")

print("="*80 + "\n")

def print_stats(header, stat):
    acc = stat["correct"] / stat["total"] if stat["total"] > 0 else 0
    rec = stat["recall"] / stat["total"] if stat["total"] > 0 else 0
    python_pct = stat["had_python"] / stat["total"] if stat["total"] > 0 else 0
    print(f"{header} | Accuracy: {acc:.3f} | Recall: {rec:.3f} | Python: {python_pct:.1%} | N: {stat['total']}")

print("="*80)
print("HYBRID RAG (No Summary) - Always LLM with Python as Context")
print("="*80)

print("\nOverall")
print_stats("ALL", stats_hybrid_no_summary["all"])

print("\nBy answer type")
print_stats("Number", stats_hybrid_no_summary["number"])

print("\nBy difficulty")
for d in sorted(stats_difficulty_hybrid_no_summary):
    print_stats(f"Difficulty {d}", stats_difficulty_hybrid_no_summary[d])

print("\n")

##############################################################
# HYBRID APPROACH: With Summary + Python as Context
##############################################################

vector_store_summary = Chroma.from_documents(
    documents=docs_summary, 
    embedding=embeddings, 
    collection_name="pl_summary_hybrid"
)

retriever_summary = vector_store_summary.as_retriever(search_kwargs={"k": 5})

template_hybrid_summary = """
!!! CRITICAL INSTRUCTION !!!
If you see "=== PYTHON CALCULATION RESULT ===" below, you MUST do the following:
1. Look for the line that says "Calculated [stat_type]: [number]"
2. Use that EXACT number in your FINAL answer
3. DO NOT count matches manually
4. DO NOT add numbers from different matchday blocks yourself
5. The Python calculation is always correct - trust it completely

You are answering questions about the Premier League 2024/25 season.

Context documents show ONE team and ONE matchday block each:
Blocks: 1-8, 9-16, 17-24, 25-32, 33-38

Rules:
1) If Python calculation provided: USE IT. Do not override.
2) If no Python: Use documents to answer
3) Output EXACTLY two lines

Output format:
FINAL: <number>
EVIDENCE: <mention Python calculation if provided>

Context:
{context}

Question:
{question}
"""

prompt_hybrid_summary = ChatPromptTemplate.from_template(template_hybrid_summary)

runnable_parallel_hybrid_summary = RunnableParallel(
    context=retriever_summary, 
    question=RunnablePassthrough()
)

chain_hybrid_summary = (
    {"context": RunnableLambda(lambda x: x["context"]),
     "question": RunnableLambda(lambda x: x["question"])}
    | prompt_hybrid_summary
    | model
    | StrOutputParser()
)

stats_hybrid_summary = {
    "all": {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    "number": {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0}
}

stats_difficulty_hybrid_summary = {
    1: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    2: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    3: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    4: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0},
    5: {"correct": 0, "total": 0, "recall": 0.0, "had_python": 0}
}

failed_questions_hybrid_summary = []

print("\n" + "="*80)
print("Starting evaluation: HYBRID WITH SUMMARY (Python + LLM)")
print("="*80)

for i, q in enumerate(questions, 1):
    print(f"Processing question {i}/{len(questions)}...", end='\r')
    
    retrieval_result = runnable_parallel_hybrid_summary.invoke(q["question"])
    retrieved_docs = retrieval_result["context"]
    
    # Always use hybrid approach
    result = answer_with_hybrid_approach(
        q["question"], 
        retrieved_docs, 
        chain_hybrid_summary
    )
    
    final_answer = result['answer']
    had_python = result['had_python_calc']
    
    predicted_label = extract_label(final_answer)
    gold_label = gold_value(q)

    # Convert to int if needed
    if predicted_label and predicted_label.isdigit():
        predicted_label = int(predicted_label)

    retrieved_ids = [d.metadata["doc_id"] for d in retrieved_docs]

    # Handle both formats (simple_test_questions.json doesn't have gold_document_ids)
    if "gold_document_ids" in q:
        gold_ids = q["gold_document_ids"]
        retrieval_hits = sum(1 for g in gold_ids if g in retrieved_ids)
        recall = retrieval_hits / len(gold_ids)
    else:
        recall = 1.0  # Default for simple test questions without gold doc IDs

    correct = (predicted_label == gold_label)

    if not correct:
        failed_questions_hybrid_summary.append({
            'question': q['question'],
            'difficulty': int(q['difficulty']),
            'predicted': predicted_label,
            'gold': gold_label,
            'had_python': had_python,
            'answer': final_answer
        })

    stats_hybrid_summary["all"]["total"] += 1
    stats_hybrid_summary["all"]["recall"] += recall
    if had_python:
        stats_hybrid_summary["all"]["had_python"] += 1
    if correct:
        stats_hybrid_summary["all"]["correct"] += 1

    answer_type = q["answer_type"]
    stats_hybrid_summary[answer_type]["total"] += 1
    stats_hybrid_summary[answer_type]["recall"] += recall
    if had_python:
        stats_hybrid_summary[answer_type]["had_python"] += 1
    if correct:
        stats_hybrid_summary[answer_type]["correct"] += 1

    difficulty = int(q["difficulty"])
    stats_difficulty_hybrid_summary[difficulty]["total"] += 1
    stats_difficulty_hybrid_summary[difficulty]["recall"] += recall
    if had_python:
        stats_difficulty_hybrid_summary[difficulty]["had_python"] += 1
    if correct:
        stats_difficulty_hybrid_summary[difficulty]["correct"] += 1

print("\n" + "="*80)
print("Failed Questions Analysis - HYBRID WITH SUMMARY")
print("="*80)
print(f"Total failed: {len(failed_questions_hybrid_summary)} out of {len(questions)}")

for difficulty in [1, 2, 3, 4, 5]:
    failed_at_diff = [f for f in failed_questions_hybrid_summary if f['difficulty'] == difficulty]
    print(f"\n--- Difficulty {difficulty}: {len(failed_at_diff)} failed ---")

    for i, f in enumerate(failed_at_diff[:3], 1):
        print(f"\n{i}. {f['question']}")
        print(f"   Predicted: {f['predicted']} | Gold: {f['gold']} | Had Python: {f['had_python']}")
        if len(f['answer']) > 150:
            print(f"   Answer: {f['answer'][:150]}...")
        else:
            print(f"   Answer: {f['answer']}")

print("="*80 + "\n")

print("="*80)
print("HYBRID RAG (With Summary) - Always LLM with Python as Context")
print("="*80)

print("\nOverall")
print_stats("ALL", stats_hybrid_summary["all"])

print("\nBy answer type")
print_stats("Number", stats_hybrid_summary["number"])

print("\nBy difficulty")
for d in sorted(stats_difficulty_hybrid_summary):
    print_stats(f"Difficulty {d}", stats_difficulty_hybrid_summary[d])

print("\n")

##############################################################
#test questions

extra_questions = [
    "How many wins did Chelsea FC have in matchdays 1-8?",
    "How many goals did Manchester United score the whole season?",
    "What was the total number of points earned by Arsenal FC in matchdays 9-16?",
    "How many losses did Liverpool FC have in matchdays 17-24?",
    "What is the goal difference for Tottenham Hotspur in matchdays 25-32?"
]

print("\n" + "="*80)
print("EXTRA TEST QUESTIONS (no scoring)")
print("="*80)

for qtext in extra_questions:
    retrieval_result = runnable_parallel_hybrid_no_summary.invoke(qtext)
    retrieved_docs = retrieval_result["context"]

    result = answer_with_hybrid_approach(qtext, retrieved_docs, chain_hybrid_no_summary)


    print("\nQUESTION:", qtext)
    print(result["answer"])
    print("Had Python:", result["had_python_calc"])
    print("Retrieved doc_ids:", [d.metadata["doc_id"] for d in retrieved_docs])