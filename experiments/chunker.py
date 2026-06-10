from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

loader = PyPDFLoader(
    "data/b-tech-mtech-ntegrated-2021-regulations.pdf"
)

documents = loader.load()

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

chunks = splitter.split_documents(documents)

print(f"Original documents: {len(documents)}")
print(f"Total chunks: {len(chunks)}")

print("\n===== CHUNK 1 =====\n")
print(chunks[0].page_content)

print("\n===== CHUNK 2 =====\n")
print(chunks[1].page_content)