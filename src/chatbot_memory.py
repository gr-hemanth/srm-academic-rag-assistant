from pathlib import Path
import os

from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# Load .env from project root
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
# Load LLM
# =========================

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

# =========================
# Conversation Memory
# =========================

chat_history = []

print("\nSRM Academic Assistant (Memory Version)")
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

Context:
{context}

Current Question:
{query}

Rules:
1. Answer ONLY from the provided context.
2. Use previous conversation to understand follow-up questions.
3. If the answer is not found in the context, say:
   "I could not find this information in the provided documents."
4. Be concise and factual.

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

    pages = sorted(
        set(doc.metadata["page"] + 1 for doc in results)
    )

    print("\nSources:")
    print(", ".join(f"Page {page}" for page in pages))
    print()