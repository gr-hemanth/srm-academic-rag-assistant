import json
import re
import sys
import time
from pathlib import Path
from statistics import mean

from langchain_groq import ChatGroq

# =========================
# Project paths and imports
# =========================

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
EXPERIMENTS_DIR = PROJECT_ROOT / "experiments"

sys.path.insert(0, str(SRC_DIR))

from rag import get_answer

BENCHMARK_FILE = EXPERIMENTS_DIR / "benchmark_questions.json"
PROGRESS_FILE = EXPERIMENTS_DIR / "benchmark_75_progress.json"
FINAL_RESULTS_FILE = EXPERIMENTS_DIR / "benchmark_75_final_results.json"

# =========================
# Judge model
# =========================

judge = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# =========================
# Load benchmark
# =========================

with open(BENCHMARK_FILE, "r", encoding="utf-8") as file:
    benchmark = json.load(file)

# =========================
# Resume previous progress
# =========================

if PROGRESS_FILE.exists():
    with open(PROGRESS_FILE, "r", encoding="utf-8") as file:
        results = json.load(file)

    print(f"\nResuming evaluation. Completed: {len(results)} questions.\n")
else:
    results = []
    print("\nStarting new evaluation.\n")

completed_questions = {
    result["question"]
    for result in results
}

# =========================
# Helper functions
# =========================

def extract_score(text, metric):
    pattern = rf"{metric}\s*:\s*([0-9]*\.?[0-9]+)"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        score = float(match.group(1))
        return max(0.0, min(1.0, score))

    return 0.0


def save_progress():
    with open(PROGRESS_FILE, "w", encoding="utf-8") as file:
        json.dump(results, file, indent=2, ensure_ascii=False)


def print_final_report():
    if not results:
        print("\nNo completed results available.")
        return

    correctness_scores = [
        result["metrics"]["correctness"]
        for result in results
    ]

    faithfulness_scores = [
        result["metrics"]["faithfulness"]
        for result in results
    ]

    recall_scores = [
        result["metrics"]["context_recall"]
        for result in results
    ]

    precision_scores = [
        result["metrics"]["context_precision"]
        for result in results
    ]

    hallucination_scores = [
        result["metrics"]["hallucination"]
        for result in results
    ]

    passed = sum(
        1
        for score in correctness_scores
        if score >= 0.7
    )

    pass_rate = (passed / len(results)) * 100

    composite_score = mean([
        mean(correctness_scores),
        mean(faithfulness_scores),
        mean(recall_scores),
        mean(precision_scores)
    ])

    print("\n" + "=" * 60)
    print("FINAL REPORT")
    print("=" * 60)

    print(f"Questions Evaluated : {len(results)} / {len(benchmark)}")
    print(f"Pass Rate           : {pass_rate:.2f}%")
    print(f"Correctness         : {mean(correctness_scores):.3f}")
    print(f"Faithfulness        : {mean(faithfulness_scores):.3f}")
    print(f"Context Recall      : {mean(recall_scores):.3f}")
    print(f"Context Precision   : {mean(precision_scores):.3f}")
    print(f"Hallucination Rate  : {mean(hallucination_scores):.3f}")
    print(f"Composite Score     : {composite_score:.3f}")

    if len(results) < len(benchmark):
        print("\nSTATUS: Partial run. Rerun later to continue.")
    else:
        print("\nSTATUS: Complete benchmark run.")

    print("=" * 60)


# =========================
# Evaluation loop
# =========================

for index, item in enumerate(benchmark, start=1):
    question = item["question"]
    expected = item["expected_answer"]

    if question in completed_questions:
        print(f"Skipping Question {index} — already completed.")
        continue

    print("\n" + "=" * 60)
    print(f"Question {index} of {len(benchmark)}")
    print("=" * 60)
    print("QUESTION:")
    print(question)

    try:
        actual, sources, context = get_answer(
            question,
            use_memory=False
        )

        judge_prompt = f"""
You are evaluating a RAG system.

Question:
{question}

Expected Answer:
{expected}

Actual Answer:
{actual}

Retrieved Context:
{context}

Score each metric from 0.0 to 1.0.

Rules:

Correctness:
How close the actual answer is to the expected answer.

Faithfulness:
How well the actual answer is supported by the retrieved context.

Context Recall:
Whether the retrieved context contains enough information to answer the question.

Context Precision:
How relevant the retrieved context is to the question.

Hallucination:
0.0 = no hallucination
1.0 = complete hallucination

If the answer says:
"I could not find this information in the provided documents."
and the expected answer also says that information is unavailable,
then score:
Correctness: 1.00
Faithfulness: 1.00
ContextRecall: 1.00
ContextPrecision: score based on relevance
Hallucination: 0.00

Return ONLY this exact format:

Correctness: X.XX
Faithfulness: X.XX
ContextRecall: X.XX
ContextPrecision: X.XX
Hallucination: X.XX
"""

        response = judge.invoke(judge_prompt)
        evaluation = response.content

        correctness = extract_score(evaluation, "Correctness")
        faithfulness = extract_score(evaluation, "Faithfulness")
        recall = extract_score(evaluation, "ContextRecall")
        precision = extract_score(evaluation, "ContextPrecision")
        hallucination = extract_score(evaluation, "Hallucination")

        result = {
            "number": index,
            "question": question,
            "expected_answer": expected,
            "actual_answer": actual,
            "sources": [
                {
                    "document": source_pdf,
                    "page": page
                }
                for source_pdf, page in sources
            ],
            "metrics": {
                "correctness": correctness,
                "faithfulness": faithfulness,
                "context_recall": recall,
                "context_precision": precision,
                "hallucination": hallucination
            }
        }

        results.append(result)
        completed_questions.add(question)

        # Save immediately after every completed question
        save_progress()

        print("\nACTUAL:")
        print(actual)

        print("\nMETRICS")
        print(f"Correctness      : {correctness:.2f}")
        print(f"Faithfulness     : {faithfulness:.2f}")
        print(f"Context Recall   : {recall:.2f}")
        print(f"Context Precision: {precision:.2f}")
        print(f"Hallucination    : {hallucination:.2f}")

        print(f"\nSaved progress: {len(results)} / {len(benchmark)}")

        # Reduces request bursts and makes rate limits less likely
        time.sleep(3)

    except Exception as error:
        save_progress()

        print("\n" + "=" * 60)
        print(f"STOPPED AT QUESTION {index}")
        print("=" * 60)
        print(error)
        print("\nProgress was saved.")
        print("Run the same command later to continue from this question.")
        break

# =========================
# Save final/partial results
# =========================

save_progress()

with open(FINAL_RESULTS_FILE, "w", encoding="utf-8") as file:
    json.dump(results, file, indent=2, ensure_ascii=False)

print_final_report()