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

llm = ChatGroq(
    model="llama-3.3-70b-versatile"
)

test_questions = [
    "What is the attendance requirement?",
    "What is attendance condonation?",
    "Who can suspend a student?",
    "What is attendance shortage?",
    "How many credits are required for the B.Tech degree?"
]

for question in test_questions:

    print("\n" + "=" * 70)
    print("QUESTION:")
    print(question)

    results = vector_store.similarity_search(
        question,
        k=5
    )

    context = "\n\n".join(
        doc.page_content for doc in results
    )

    prompt = f"""
You are an SRM Academic Assistant.

Answer ONLY from the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    response = llm.invoke(prompt)

    print("\nANSWER:")
    print(response.content)

    pages = sorted(
        set(doc.metadata["page"] + 1 for doc in results)
    )

    print("\nPAGES:")
    print(pages)