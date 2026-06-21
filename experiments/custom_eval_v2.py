import json
import re
from statistics import mean

from langchain_groq import ChatGroq
from rag import get_answer

judge = ChatGroq(
    model="llama-3.3-70b-versatile"
)

with open(
    "src/benchmark_questions.json",
    "r",
    encoding="utf-8"
) as f:
    benchmark = json.load(f)

correctness_scores = []
faithfulness_scores = []
recall_scores = []
precision_scores = []
hallucination_scores = []

passed = 0

print("\nRunning Evaluation...\n")


def extract_score(text, metric):

    pattern = rf"{metric}\s*:\s*([0-9]*\.?[0-9]+)"

    match = re.search(
        pattern,
        text,
        re.IGNORECASE
    )

    if match:
        return float(match.group(1))

    return 0.0


for idx, item in enumerate(
    benchmark,
    start=1
):

    question = item["question"]
    expected = item["expected_answer"]

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
How close actual answer is to expected answer.

Faithfulness:
How well actual answer is supported by context.

Context Recall:
Did retrieved context contain enough information?

Context Precision:
How relevant is retrieved context?

Hallucination:
0.0 = no hallucination
1.0 = complete hallucination

IMPORTANT:
If the model says
"I could not find this information"
then hallucination = 0.0

Return ONLY:

Correctness: X.XX
Faithfulness: X.XX
ContextRecall: X.XX
ContextPrecision: X.XX
Hallucination: X.XX
"""

    response = judge.invoke(
        judge_prompt
    )

    evaluation = response.content

    correctness = extract_score(
        evaluation,
        "Correctness"
    )

    faithfulness = extract_score(
        evaluation,
        "Faithfulness"
    )

    recall = extract_score(
        evaluation,
        "ContextRecall"
    )

    precision = extract_score(
        evaluation,
        "ContextPrecision"
    )

    hallucination = extract_score(
        evaluation,
        "Hallucination"
    )

    correctness_scores.append(
        correctness
    )

    faithfulness_scores.append(
        faithfulness
    )

    recall_scores.append(
        recall
    )

    precision_scores.append(
        precision
    )

    hallucination_scores.append(
        hallucination
    )

    if correctness >= 0.7:
        passed += 1

    print("=" * 60)
    print(f"Question {idx}")
    print("=" * 60)

    print("Question:")
    print(question)

    print("\nCorrectness      :", correctness)
    print("Faithfulness     :", faithfulness)
    print("Context Recall   :", recall)
    print("Context Precision:", precision)
    print("Hallucination    :", hallucination)
    print()


accuracy = (
    passed / len(benchmark)
) * 100

composite_score = mean([
    mean(correctness_scores),
    mean(faithfulness_scores),
    mean(recall_scores),
    mean(precision_scores)
])

print("\n")
print("=" * 60)
print("FINAL REPORT")
print("=" * 60)

print(
    f"Questions Evaluated : {len(benchmark)}"
)

print(
    f"Pass Rate           : {accuracy:.2f}%"
)

print(
    f"Correctness         : {mean(correctness_scores):.3f}"
)

print(
    f"Faithfulness        : {mean(faithfulness_scores):.3f}"
)

print(
    f"Context Recall      : {mean(recall_scores):.3f}"
)

print(
    f"Context Precision   : {mean(precision_scores):.3f}"
)

print(
    f"Hallucination Rate  : {mean(hallucination_scores):.3f}"
)

print(
    f"Composite Score     : {composite_score:.3f}"
)

print("=" * 60)