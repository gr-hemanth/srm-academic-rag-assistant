import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

loader = PyPDFLoader(
    "data/b-tech-mtech-ntegrated-2021-regulations.pdf"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_store = FAISS.from_documents(
    chunks,
    embedding_model
)

query = input("Ask a question: ")

results = vector_store.similarity_search(
    query,
    k=3
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
    print(f"Page: {doc.metadata['page'] + 1}")