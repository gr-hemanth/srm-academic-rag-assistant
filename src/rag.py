from pathlib import Path

from dotenv import load_dotenv
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq

load_dotenv(Path(__file__).parent.parent / ".env")

# =========================
# Models
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Knowledge Base
# =========================

vector_store = None
all_docs = None
bm25 = None


def load_knowledge_base():

    global vector_store
    global all_docs
    global bm25

    vector_store = FAISS.load_local(
        "faiss_index",
        embedding_model,
        allow_dangerous_deserialization=True
    )

    all_docs = list(
        vector_store.docstore._dict.values()
    )

    tokenized_docs = [
        doc.page_content.lower().split()
        for doc in all_docs
    ]

    bm25 = BM25Okapi(tokenized_docs)

    print(
        f"Knowledge Base Loaded: "
        f"{len(all_docs)} chunks"
    )

def get_chunk_count():
    global all_docs
    return len(all_docs) if all_docs else 0

load_knowledge_base()

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)
llm = ChatGroq(
    model="qwen/qwen3.6-27b",
    temperature=0,
    reasoning_effort="none",
)

chat_history = []

def get_answer(
    query,
    use_memory=True
):
    # -------------------------
    # Conversational Retrieval
    # -------------------------

    search_query = query

    if use_memory and len(chat_history) >= 2:
        search_query = (
            chat_history[-2]["content"]
            + " "
            + query
        )
    # -------------------------
    # Hybrid Retrieval
    # -------------------------

    faiss_results = vector_store.similarity_search(
        search_query,
        k=20
    )

    query_tokens = search_query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[-20:][::-1]

    bm25_results = [
        all_docs[idx]
        for idx in top_indices
    ]

    results = []
    seen = set()

    for doc in faiss_results + bm25_results:

        if doc.page_content not in seen:
            seen.add(doc.page_content)
            results.append(doc)

    # -------------------------
    # Reranking
    # -------------------------

    pairs = [
        (search_query, doc.page_content)
        for doc in results
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True
    )

    # Keep top 5 reranked chunks
    results = [
        doc
        for score, doc in ranked_docs[:5]
    ]

    context = "\n\n".join(
        doc.page_content
        for doc in results
    )

    # -------------------------
    # Chat History
    # -------------------------

    history_text = ""

    if use_memory:
        for message in chat_history:
            history_text += (
                f"{message['role']}: "
                f"{message['content']}\n"
            )

    prompt = f"""
You are the SRM Academic Assistant.

Previous Conversation:
{history_text}

Rules:
1. Answer ONLY using the provided context.
2. Use exact values, numbers, and terminology from the context.
3. Do not combine information from different programs unless the question explicitly asks for a comparison.
4. If the answer is not present in the context, reply exactly:
   "I could not find this information in the provided documents."
5. Be concise, factual, and professional.
6. Mention regulation numbers whenever they are available.
7. Use previous conversation only to understand follow-up questions.
8. Do not make assumptions or infer information that is not explicitly present in the context.

Formatting:
- Use Markdown sparingly.
- Bold only important values such as:
  - percentages
  - credit counts
  - grades
  - regulation numbers
- Do NOT bold complete sentences.
- Do NOT bold long headings.
- Use short paragraphs.
- If multiple points are required, use bullet points.
- Do not use emojis.


Context:
{context}

Current Question:
{query}

Answer:
"""

    response = llm.invoke(prompt)

    answer = response.content
    import re
    answer = re.sub(r"\*{1,3}(.*?)\*{1,3}", r"\1", answer)

    if use_memory:

        chat_history.append(
            {
                "role": "user",
                "content": query
            }
        )

        chat_history.append(
            {
                "role": "assistant",
                "content":answer
            }
        )

    sources = sorted(
        set(
            (
                doc.metadata.get("source_pdf", "Unknown"),
                doc.metadata["page"] + 1
            )
            for doc in results
        )
    )

    return answer, sources, context