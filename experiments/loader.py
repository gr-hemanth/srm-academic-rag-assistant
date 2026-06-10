from langchain_community.document_loaders import PyPDFLoader

loader = PyPDFLoader(
    "data/b-tech-mtech-ntegrated-2021-regulations.pdf"
)

documents = loader.load()

print(f"Total pages loaded: {len(documents)}")

print("\n===== PAGE 1 =====\n")
print(documents[0].page_content[:300])

print("\n===== PAGE 10 =====\n")
print(documents[9].page_content[:300])

print("\n===== PAGE 20 =====\n")
print(documents[19].page_content[:300])