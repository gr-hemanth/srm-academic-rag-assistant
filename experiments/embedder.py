from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer

loader = PyPDFLoader(
    "data/b-tech-mtech-ntegrated-2021-regulations.pdf"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

model = SentenceTransformer("all-MiniLM-L6-v2")

embedding = model.encode(chunks[0].page_content)

print("Embedding length:", len(embedding))

print("\nFirst 10 values:")
print(embedding[:10])