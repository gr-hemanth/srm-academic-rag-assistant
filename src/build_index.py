import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings


def build_index():

    all_documents = []

    for file_name in os.listdir("data"):

        if file_name.endswith(".pdf"):

            pdf_path = os.path.join(
                "data",
                file_name
            )

            loader = PyPDFLoader(pdf_path)

            documents = loader.load()

            for doc in documents:
                doc.metadata["source_pdf"] = file_name

            all_documents.extend(documents)


    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )

    chunks = splitter.split_documents(
        all_documents
    )

    embedding_model = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    print(
        f"Loaded {len(all_documents)} pages"
    )

    print(
        f"Created {len(chunks)} chunks"
    )

    vector_store = FAISS.from_documents(
        chunks,
        embedding_model
    )

    vector_store.save_local("faiss_index")

    print("FAISS index saved successfully!")


if __name__ == "__main__":

    build_index()