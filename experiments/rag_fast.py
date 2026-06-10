from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.load_local(
    "faiss_index",
    embedding_model,
    allow_dangerous_deserialization=True
)

query = input("Ask a question: ")

results = vector_store.similarity_search(
    query,
    k=10
)

context = "\n\n".join(
    doc.page_content for doc in results
)

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

prompt = f"""
Answer the question using ONLY the provided context.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)

print("\n===== ANSWER =====\n")
print(response.content)

print("\n===== SOURCES =====\n")

for doc in results:
    print(f"Page {doc.metadata['page'] + 1}")