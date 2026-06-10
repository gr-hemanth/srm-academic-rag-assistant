# SRM Academic RAG Assistant

A Retrieval-Augmented Generation (RAG) chatbot built using:

- LangChain
- FAISS
- Sentence Transformers
- Groq

## Features

- PDF Question Answering
- Semantic Search
- Source Attribution
- Evaluation Pipeline

## Architecture

PDF
→ Chunking
→ Embeddings
→ FAISS
→ Retrieval
→ Groq LLM
→ Answer