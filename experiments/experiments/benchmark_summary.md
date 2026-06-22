# SRM Academic Assistant — V1 Evaluation

## Benchmark
- Questions evaluated: 74
- Evaluation mode: Independent questions (`use_memory=False`)
- Retrieval: FAISS + BM25 hybrid retrieval
- Reranking: `cross-encoder/ms-marco-MiniLM-L-6-v2`
- Generator and judge: Groq `llama-3.3-70b-versatile`

## Results

| Metric | Score |
|---|---:|
| Pass Rate | 89.19% |
| Correctness | 0.852 |
| Faithfulness | 0.984 |
| Context Recall | 0.956 |
| Context Precision | 0.929 |
| Hallucination Rate | 0.049 |
| Composite Score | 0.930 |

## Interpretation
The system showed high retrieval coverage and strong grounding. Most generated answers were supported by retrieved context, while hallucination remained low. Remaining errors are mainly expected in ambiguous, missing-information, or fine-grained policy questions.