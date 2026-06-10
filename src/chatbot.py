from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

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
# Load LLM
# =========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

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

    results = vector_store.similarity_search(
        query,
        k=10
    )

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    # -------------------------
    # Prompt
    # -------------------------

    prompt = f"""
You are an SRM Academic Assistant.

Rules:
1. Answer ONLY using the provided context.
2. Use exact values and numbers from the context.
3. Do not combine information from different programs unless the question asks for comparison.
4. If the answer is not present in the context, say:
   "I could not find this information in the provided documents."
5. Be concise and factual.
6. Mention regulation numbers if available.

Context:
{context}

Question:
{query}

Answer:
"""

    # -------------------------
    # LLM Response
    # -------------------------

    response = llm.invoke(prompt)

    print("\nAssistant:")
    print(response.content)

    # -------------------------
    # Sources
    # -------------------------

    pages = sorted(
        set(doc.metadata["page"] + 1 for doc in results)
    )

    print("\nSources:")
    print(", ".join(f"Page {page}" for page in pages))
    print()