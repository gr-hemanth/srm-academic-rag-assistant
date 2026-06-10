from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings

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

query = "What is the attendance requirement?"

results = vector_store.similarity_search(
    query,
    k=3
)

for i, doc in enumerate(results, start=1):
    print(f"\n===== RESULT {i} =====\n")
    print(doc.page_content[:500])