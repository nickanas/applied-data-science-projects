# =====================================================================================
# Hybrid RAG system for answering numerical questions about the Premier League 2024/25.
#
# The pipeline combines two sources of truth:
#   1. A vector-retrieval RAG component that surfaces the relevant match documents.
#   2. A deterministic Python calculation layer that computes the exact statistic
#      (points, wins, goals, etc.) for the team and matchday range in the question.
#
# The Python result is injected into the LLM prompt as authoritative context, so the
# model reports an exact figure instead of attempting (and often failing) the arithmetic.
#
# The script builds the document store, evaluates two retrieval variants (with and
# without per-block summaries) against a 100-question benchmark, and reports accuracy /
# retrieval recall broken down by answer type and difficulty.
# =====================================================================================

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

##############################################################
# Load and normalise the raw season data
##############################################################

# Source data follows the openfootball JSON schema (one entry per fixture).
data = json.load(open("en.1.json"))
season = data.get("name", "Premier League 2024/25")
matches = data["matches"]


# Safely cast a value to int, returning `default` if it can't be parsed.
def to_int(x, default=None):
    try:
        return int(x)
    except (TypeError, ValueError):
        return default


# Flatten each fixture into a single record with explicit half-time / full-time scores.
# Scores may be missing or malformed in the raw data, so each field is parsed defensively.
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

# Re-index fixtures by team. Each fixture appears twice (once per side) so that every
# team has a complete, team-centric view of its season with for/against scores.
team_matches = {}

for t in structured_matches:
    # `matchday` is a free-text round label (e.g. "Matchday 5"); pull out the numeric value.
    matchday_integers = [int(s) for s in t["matchday"].split() if s.isdigit()]
    matchday_integer = matchday_integers[0]

    # Home-team perspective.
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

    # Away-team perspective (for/against scores are swapped).
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

# Order each team's fixtures chronologically (matchday, then time, then date).
for team in team_matches:
    team_matches[team].sort(key=lambda x: x["matchday number"])
    team_matches[team].sort(key=lambda x: x["time"])
    team_matches[team].sort(key=lambda x: x["date"])

##############################################################
# Build the retrieval documents
##############################################################

# The season is split into five matchday blocks. Each (team, block) pair becomes one
# retrieval document, which keeps documents small and topically focused.
blocks = [(1, 8), (9, 16), (17, 24), (25, 32), (33, 38)]


# Build one LangChain Document per (team, matchday block). Each document lists every
# fixture in the block. When `summary` is True an aggregated stats footer (points,
# wins/draws/losses, goals) is appended, giving the retriever a numeric summary to match.
def build_documents(summary: bool):
    docs = []

    for team in team_matches:
        matches = team_matches[team]

        for block in blocks:
            first_md, last_md = block

            # Collect the fixtures that fall inside this block.
            matches_block = []
            for m in matches:
                if first_md <= m["matchday number"] <= last_md:
                    matches_block.append(m)

            # Aggregate block-level statistics (only used in the summary footer below).
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

            # Document header: team, block range and season metadata.
            document_text = f"""
TEAM: {team}
MATCHDAYS: {first_md}-{last_md}\n\n
League: Premier League
SEASON: {season}"""

            # Per-fixture lines.
            for m in matches_block:
                document_text += (
                    f"{m['round']}\n"
                    f"{m['date']} {m['time']}\n"
                    f"{m['team']} ({m['venue']}) vs {m['opponent']}\n"
                    f"Half time score: {m['ht_for']}-{m['ht_against']}\n"
                    f"Full time score: {m['ft_for']}-{m['ft_against']}\n\n"
                )

            # Optional aggregated footer.
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


# Two document variants: one with the stats footer, one without.
docs_summary = build_documents(summary=True)
docs_no_summary = build_documents(summary=False)

##############################################################
# Deterministic Python calculation layer
##############################################################
# These helpers compute the exact answer for a question so the LLM never has to do the
# arithmetic itself. This is the key to reliable numeric answers.


# Return all fixtures for a team within an inclusive matchday range.
def extract_matches_for_team(team_name, start_md, end_md):
    if team_name not in team_matches:
        return []

    matches = team_matches[team_name]
    filtered = [
        m for m in matches
        if start_md <= m['matchday number'] <= end_md
    ]
    return filtered


# Compute aggregated statistics for a team across one or more matchday ranges.
# `ranges` is a list of inclusive (start, end) tuples. Returns a dict of points, goals,
# goal difference and W/D/L counts, or None if no fixtures match or a score is missing.
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
            return None  # Incomplete data: refuse to report a partial figure.

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


# Resolve which team a question refers to. Generates aliases for each team (e.g. dropping
# "FC"/"AFC", plus short forms like "man city") and matches them as whole words. When
# several teams match, the longest alias wins to avoid false positives from substrings.
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

        # Hand-crafted short forms for the two Manchester clubs.
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

    # Prefer the longest matching alias (most specific team name).
    candidates.sort(reverse=True)
    return candidates[0][1]


# Extract the matchday range(s) referenced by a question. Recognises explicit ranges
# ("matchdays 9-16", "9 to 16"), bare day-pairs, single matchdays and whole-season
# phrasing. Overlapping/adjacent ranges are merged; defaults to the full season (1-38).
def parse_matchday_ranges(question: str):
    q = question.lower()
    ranges = []

    # "matchday(s) 9-16" or "matchday(s) 9 to 16".
    for a, b in re.findall(r"matchdays?\s+(\d+)\s*(?:-|to)\s*(\d+)", q):
        ranges.append((int(a), int(b)))

    # Bare day-pairs like "9-16", constrained to the valid 1-38 range.
    for a, b in re.findall(r"\b(\d{1,2})\s*-\s*(\d{1,2})\b", q):
        a, b = int(a), int(b)
        if 1 <= a <= 38 and 1 <= b <= 38:
            ranges.append((a, b))

    # Single matchday, e.g. "matchday 5".
    for a in re.findall(r"\bmatchday\s+(\d+)\b", q):
        md = int(a)
        ranges.append((md, md))

    # Whole-season phrasing overrides everything else.
    if "whole season" in q or "entire season" in q or "in the season" in q:
        return [(1, 38)]

    if not ranges:
        return [(1, 38)]

    # Merge overlapping or adjacent ranges into a minimal set.
    ranges.sort()
    merged = []
    for s, e in ranges:
        if not merged or s > merged[-1][1] + 1:
            merged.append([s, e])
        else:
            merged[-1][1] = max(merged[-1][1], e)
    return [(s, e) for s, e in merged]


# Parse a question into the components needed for a deterministic calculation: the team,
# the matchday window and the statistic being asked about. Returns stat_type=None when the
# question is out of scope for the calculator (e.g. head-to-head, home/away splits or
# half-time queries), which signals the pipeline to fall back to pure retrieval.
def parse_question_for_calculation(question):
    question_lower = question.lower()

    # Phrases that imply a comparison or sub-filter the simple calculator can't handle.
    # When present, return team-only with no stat_type so we defer to the LLM/documents.
    blocked_phrases = [
        "against", "vs", "versus",
        "home game", "away game", "both games",
        "in the games", "in both games",
        "than", "more than", "difference between", "how many more"
    ]

    if any(p in question_lower for p in blocked_phrases):
        return {"team": find_team_in_question(question_lower, team_matches.keys()), "start_md": 1, "end_md": 38, "stat_type": None}

    team_found = find_team_in_question(question_lower, team_matches.keys())

    # Use parse_matchday_ranges instead of a manual regex because it handles multiple ranges correctly.
    ranges = parse_matchday_ranges(question)

    # Collapse the parsed ranges into a single start/end window for backwards compatibility.
    if ranges:
        start_md = ranges[0][0]  # Start of first range
        end_md = ranges[-1][1]   # End of last range
    else:
        start_md = 1
        end_md = 38

    # Half-time questions aren't supported by the calculator; defer to retrieval.
    if any(phrase in question_lower for phrase in ['first half', 'half time', 'halftime', 'half-time', ' ht ', 'half ']):
        return {
            "team": team_found,
            "start_md": start_md,
            "end_md": end_md,
            "stat_type": None
        }

    # Map question wording to the statistic key returned by calculate_team_stats().
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
# Hybrid Python + LLM orchestration
##############################################################


# Concatenate retrieved documents into a single context string.
def format_docs(docs):
    return "\n\n".join(doc.page_content for doc in docs)


# Compute the exact answer for a question and render it as a prompt context block.
# Returns an empty string when the question can't be resolved to a (team, stat_type) pair
# or the calculation fails, in which case the LLM relies on retrieval alone.
def get_python_calculation_context(question):
    parsed = parse_question_for_calculation(question)

    if not (parsed["team"] and parsed["stat_type"]):
        return ""

    ranges = parse_matchday_ranges(question)
    stats = calculate_team_stats(parsed["team"], ranges)

    if not stats:
        return ""

    result = stats[parsed["stat_type"]]

    # Surface the headline figure plus the full stat line so the LLM can cross-check it.
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


# Answer a question with the hybrid strategy: always run the LLM, but prepend the
# deterministic Python result (when available) ahead of the retrieved documents.
# Returns the LLM answer plus flags indicating whether a Python calculation was used.
def answer_with_hybrid_approach(question, retrieved_docs, chain):
    # Python calculation context (empty string if not applicable).
    python_context = get_python_calculation_context(question)

    # Retrieved documents rendered as context.
    docs_context = format_docs(retrieved_docs)

    # Place the authoritative Python result first so it anchors the model's answer.
    if python_context:
        enhanced_context = python_context + "\n\n" + docs_context
    else:
        enhanced_context = docs_context

    # Diagnostics: surface what the calculator produced for the current question.
    print(f"\n{'='*60}")
    print(f"DEBUG - Question: {question[:80]}")
    print(f"DEBUG - Python context present: {bool(python_context)}")
    if python_context:
        # Echo just the computed figure from the Python context block.
        match = re.search(r'Calculated \w+: (\d+)', python_context)
        if match:
            print(f"DEBUG - Python computed: {match.group(1)}")
    print(f"DEBUG - Context length: {len(enhanced_context)} chars")

    # Invoke the LLM with the combined context.
    llm_result = chain.invoke({
        "context": enhanced_context,
        "question": question
    })

    print(f"DEBUG - LLM output: {llm_result[:100]}...")
    print(f"{'='*60}\n")

    return {
        "answer": llm_result,
        "had_python_calc": bool(python_context),
        "python_context": python_context
    }

##############################################################
# Load the evaluation benchmark
##############################################################

with open("questions100.json") as f:
    questions = json.load(f)

# Lookup from document id to Document, used when inspecting gold documents per question.
doc_by_id = {doc.metadata["doc_id"]: doc for doc in docs_no_summary}


# Debug helper: print a benchmark question alongside its gold documents.
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
# Initialise the LLM and embedding models
##############################################################

# Mathstral is a maths-focused model; deterministic decoding (do_sample=False) keeps
# answers reproducible across evaluation runs.
model = HuggingFacePipeline.from_model_id(
    model_id="mistralai/Mathstral-7B-v0.1",
    task="text-generation",
    pipeline_kwargs={"return_full_text": False, "do_sample": False},
)

embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-mpnet-base-v2")


# Pull the predicted value from the model's `FINAL: <value>` output line.
def extract_label(answer: str):
    text = re.search(r"FINAL:\s*(.*)", answer)
    if not text:
        return None
    label = text.group(1).strip()
    return label


# Return the gold answer, normalising across the two supported benchmark schemas.
def gold_value(q):
    # simple_test_questions.json format (has 'gold_answer').
    if "gold_answer" in q:
        return int(q["gold_answer"])

    # questions100.json format.
    if q.get("answer_type") == "number":
        return int(q["answer"]["value"])
    return q["answer"]["value"].strip()

##############################################################
# EVALUATION 1 - Hybrid RAG without per-block summaries
##############################################################

# Index the no-summary documents and expose a top-5 retriever.
vector_store_no_summary = Chroma.from_documents(
    documents=docs_no_summary,
    embedding=embeddings,
    collection_name="pl_no_summary_hybrid"
)

retriever_no_summary = vector_store_no_summary.as_retriever(search_kwargs={"k": 5})

# Prompt deliberately foregrounds the Python result and forbids the model from doing its
# own arithmetic, then constrains output to a fixed two-line FINAL/EVIDENCE format.
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

# Retrieve context and pass the question through unchanged, in parallel.
runnable_parallel_hybrid_no_summary = RunnableParallel(
    context=retriever_no_summary,
    question=RunnablePassthrough()
)

# Full chain: context/question -> prompt -> model -> plain-text output.
chain_hybrid_no_summary = (
    {"context": RunnableLambda(lambda x: x["context"]),
     "question": RunnableLambda(lambda x: x["question"])}
    | prompt_hybrid_no_summary
    | model
    | StrOutputParser()
)

# Accumulators: overall + by answer type, each tracking correctness, recall and
# how often the Python calculator contributed.
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

    # Retrieve, then answer with the hybrid (retrieval + Python) approach.
    retrieval_result = runnable_parallel_hybrid_no_summary.invoke(q["question"])
    retrieved_docs = retrieval_result["context"]

    result = answer_with_hybrid_approach(
        q["question"],
        retrieved_docs,
        chain_hybrid_no_summary
    )

    final_answer = result['answer']
    had_python = result['had_python_calc']

    predicted_label = extract_label(final_answer)
    gold_label = gold_value(q)

    # Compare as integers when the prediction is numeric.
    if predicted_label and predicted_label.isdigit():
        predicted_label = int(predicted_label)

    retrieved_ids = [d.metadata["doc_id"] for d in retrieved_docs]

    # Retrieval recall: fraction of gold documents present in the retrieved set.
    # simple_test_questions.json has no gold doc ids, so default recall to 1.0 there.
    if "gold_document_ids" in q:
        gold_ids = q["gold_document_ids"]
        retrieval_hits = sum(1 for g in gold_ids if g in retrieved_ids)
        recall = retrieval_hits / len(gold_ids)
    else:
        recall = 1.0

    correct = (predicted_label == gold_label)

    # Record misses for the per-difficulty error analysis printed below.
    if not correct:
        failed_questions_hybrid_no_summary.append({
            'question': q['question'],
            'difficulty': int(q['difficulty']),
            'predicted': predicted_label,
            'gold': gold_label,
            'had_python': had_python,
            'answer': final_answer
        })

    # Update overall accumulators.
    stats_hybrid_no_summary["all"]["total"] += 1
    stats_hybrid_no_summary["all"]["recall"] += recall
    if had_python:
        stats_hybrid_no_summary["all"]["had_python"] += 1
    if correct:
        stats_hybrid_no_summary["all"]["correct"] += 1

    # Update by-answer-type accumulators.
    answer_type = q["answer_type"]
    stats_hybrid_no_summary[answer_type]["total"] += 1
    stats_hybrid_no_summary[answer_type]["recall"] += recall
    if had_python:
        stats_hybrid_no_summary[answer_type]["had_python"] += 1
    if correct:
        stats_hybrid_no_summary[answer_type]["correct"] += 1

    # Update by-difficulty accumulators.
    difficulty = int(q["difficulty"])
    stats_difficulty_hybrid_no_summary[difficulty]["total"] += 1
    stats_difficulty_hybrid_no_summary[difficulty]["recall"] += recall
    if had_python:
        stats_difficulty_hybrid_no_summary[difficulty]["had_python"] += 1
    if correct:
        stats_difficulty_hybrid_no_summary[difficulty]["correct"] += 1

# Error analysis: up to three example failures per difficulty level.
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


# Print accuracy, retrieval recall and Python-usage rate for one accumulator.
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
# EVALUATION 2 - Hybrid RAG with per-block summaries
##############################################################
# Identical pipeline to Evaluation 1, but the documents include the aggregated stats
# footer. This isolates the effect of summary text on retrieval and answer quality.

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

    result = answer_with_hybrid_approach(
        q["question"],
        retrieved_docs,
        chain_hybrid_summary
    )

    final_answer = result['answer']
    had_python = result['had_python_calc']

    predicted_label = extract_label(final_answer)
    gold_label = gold_value(q)

    if predicted_label and predicted_label.isdigit():
        predicted_label = int(predicted_label)

    retrieved_ids = [d.metadata["doc_id"] for d in retrieved_docs]

    if "gold_document_ids" in q:
        gold_ids = q["gold_document_ids"]
        retrieval_hits = sum(1 for g in gold_ids if g in retrieved_ids)
        recall = retrieval_hits / len(gold_ids)
    else:
        recall = 1.0

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
# Ad-hoc sanity checks (not scored)
##############################################################
# A few hand-written questions for a quick qualitative look at the pipeline's output.

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
