from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq
from pathlib import Path
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")

# =========================
# Load Embedding Model
# =========================

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# =========================
# Load Saved FAISS Index
# =========================

vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)
# =========================
# Build BM25 Index
# =========================

all_docs = list(
    vector_store.docstore._dict.values()
)

tokenized_docs = [
    doc.page_content.lower().split()
    for doc in all_docs
]

bm25 = BM25Okapi(tokenized_docs)


print(f"BM25 loaded with {len(all_docs)} chunks")
# =========================
# Load LLM
# =========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)
# =========================
# Load Reranker
# =========================

reranker = CrossEncoder(
    "cross-encoder/ms-marco-MiniLM-L-6-v2"
)

print("Reranker loaded")
# =========================
# Conversation Memory
# =========================

chat_history = []

print("\nSRM Academic Assistant")
print("Type 'exit' to quit.\n")

# =========================
# Chat Loop
# =========================

while True:

    query = input("You: ").strip()

    if query.lower() == "exit":
        print("\nGoodbye!")
        break

    # -------------------------
    # Retrieval
    # -------------------------

    # FAISS Results
    faiss_results = vector_store.similarity_search(
        query,
        k=5
    )

    # BM25 Results
    query_tokens = query.lower().split()

    scores = bm25.get_scores(query_tokens)

    top_indices = scores.argsort()[-5:][::-1]

    bm25_results = [    
    all_docs[idx]
    for idx in top_indices
]

    # Merge Results (remove duplicates)

    seen = set()
    results = []

    for doc in faiss_results + bm25_results:    

        if doc.page_content not in seen:
            seen.add(doc.page_content)
            results.append(doc)
    # -------------------------
    # Reranking
    # -------------------------

    pairs = [
        (query, doc.page_content)
        for doc in results
    ]

    scores = reranker.predict(pairs)

    ranked_docs = sorted(
        zip(scores, results),
        key=lambda x: x[0],
        reverse=True
    )

    results = [
        doc
        for score, doc in ranked_docs[:8]
    ]

    context = "\n\n".join(  
    doc.page_content
    for doc in results
)

    # -------------------------
    # Format Chat History
    # -------------------------

    history_text = ""

    for message in chat_history:
        history_text += (
            f"{message['role']}: "
            f"{message['content']}\n"
        )

    # -------------------------
    # Prompt
    # -------------------------

    prompt = f"""
You are an SRM Academic Assistant.

Previous Conversation:
{history_text}

Rules:
1. Answer ONLY using the provided context.
2. Use exact values and numbers from the context.
3. Do not combine information from different programs unless the question asks for comparison.
4. If the answer is not present in the context, say:
   "I could not find this information in the provided documents."
5. Be concise and factual.
6. Mention regulation numbers if available.
7. Use previous conversation to understand follow-up questions.

Context:
{context}

Current Question:
{query}

Answer:
"""

    # -------------------------
    # LLM Response
    # -------------------------

    response = llm.invoke(prompt)

    # -------------------------
    # Save Memory
    # -------------------------

    chat_history.append(
        {
            "role": "user",
            "content": query
        }
    )

    chat_history.append(
        {
            "role": "assistant",
            "content": response.content
        }
    )

    # -------------------------
    # Print Answer
    # -------------------------

    print("\nAssistant:")
    print(response.content)

    # -------------------------
    # Sources
    # -------------------------

    sources = sorted(
        set(
            (
                doc.metadata.get("source_pdf", "Unknown"),
                doc.metadata["page"] + 1
            )
            for doc in results
        )
    )

    print("\nSources:")

    for source_pdf, page in sources:
        print(f"{source_pdf} - Page {page}")

    print()