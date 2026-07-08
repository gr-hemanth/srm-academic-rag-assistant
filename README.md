# 🎓 SRM Academic RAG Assistant

> A Hybrid Retrieval-Augmented Generation (RAG) system that enables students to interact with official SRM Institute documents using natural language. The assistant retrieves relevant information from uploaded PDFs, re-ranks results using a Cross-Encoder, and generates grounded responses with document and page citations.

<p align="center">

![Python](https://img.shields.io/badge/Python-3.14.3-3776AB?style=for-the-badge&logo=python&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-Framework-1C3C3C?style=for-the-badge)
![FAISS](https://img.shields.io/badge/FAISS-Vector%20Store-0467DF?style=for-the-badge)
![Streamlit](https://img.shields.io/badge/Streamlit-Frontend-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-LLM%20Inference-F55036?style=for-the-badge)
</p>

---

## 📖 Overview

Searching through university regulations, student handbooks, and hostel policies can be time-consuming and inefficient. Students often need to navigate hundreds of pages to locate a single rule, deadline, or policy.

**SRM Academic RAG Assistant** addresses this problem by enabling users to ask questions in natural language while retrieving information directly from official university documents. Instead of relying on general-purpose language model knowledge, the assistant searches the uploaded knowledge base, retrieves the most relevant passages, re-ranks them for improved relevance, and generates responses grounded in the retrieved context.

Every response is accompanied by the corresponding source document and page reference, allowing users to verify the information directly from the original document.

---

## ✨ Key Features

- 📄 Multi-document PDF knowledge base
- 🔍 Hybrid (FAISS semantic + BM25 keyword-based) Retrieval 
- 🎯 Cross-Encoder re-ranking for improved retrieval quality
- 💬 Conversational memory for follow-up questions
- 📚 Automatic source citation with document and page references
- 📤 PDF upload with automatic knowledge base rebuilding
- ⚡ Fast inference using Groq and Llama 3.3 70B
- 🖥️ Interactive Streamlit web interface
- 📊 Custom evaluation framework for measuring RAG performance

---

# 🚀 System Architecture
```mermaid
flowchart LR

%% -------------------------------
%% Knowledge Base Construction
%% -------------------------------

subgraph KB["Knowledge Base Construction (Offline)"]

A[PDF Documents]
B[PyMuPDF Loader]
C[Recursive Text Chunking]
D[Sentence Transformer Embeddings]
E[(FAISS Vector Store)]
F[(BM25 Index)]

A --> B
B --> C
C --> D
D --> E
C --> F

end

%% -------------------------------
%% Query Processing
%% -------------------------------

subgraph QP["Query Processing (Online)"]

Q[User Query]
QE[Query Embedding]
FS[FAISS Semantic Search]
BS[BM25 Keyword Search]
M[Merge & Remove Duplicate Chunks]
R[Cross-Encoder Re-ranking]
K[Top Relevant Chunks]
H[Conversation History]
P[Prompt Construction]
L[Qwen 3.6 27B via Groq]
O[Grounded Answer<br/>+ Source Citations]

Q --> QE
QE --> FS
Q --> BS

FS --> M
BS --> M

M --> R
R --> K

H --> P
K --> P

P --> L
L --> O

end

%% -------------------------------
%% Connections
%% -------------------------------

E --> FS
F --> BS
```
---
# 📊 Evaluation

The system was evaluated using a manually curated benchmark consisting of **74 academic and hostel-related questions**.

| Metric | Score |
|---------|------:|
| Pass Rate | **89.19%** |
| Correctness | **0.852** |
| Faithfulness | **0.984** |
| Context Recall | **0.956** |
| Context Precision | **0.929** |
| Hallucination Rate | **0.049** |
| Composite Score | **0.930** |

### Evaluation Highlights

- Multi-document benchmark
- Ground-truth answer comparison
- Retrieval quality assessment
- Hallucination measurement
- Automated evaluation pipeline

---



# 🛠️ Technology Stack

| Category | Technologies |
|-----------|--------------|
| Programming Language | Python |
| Frontend | Streamlit |
| LLM | Qwen 3.6 27B (Groq) |
| AI Framework | LangChain |
| Embedding Model | sentence-transformers/all-MiniLM-L6-v2 |
| Vector Database | FAISS |
| Retrieval | Hybrid Retrieval (FAISS + BM25) |
| Re-ranking Model | Cross-Encoder (ms-marco-MiniLM-L-6-v2) |
| PDF Processing | PyMuPDF |
| Conversation | LangChain Conversation Memory |

---

# 📂 Project Structure

```text
srm-academic-rag-assistant
│
├── .venv/                            # Virtual Environment
|
├── data/                             # Knowledge base PDFs
│   ├── b-tech-mtech-integrated-2021-regulations.pdf
│   ├── hand-book-2025-26.pdf
│   └── srm-hostel-rules-2025.pdf
│
├── experiments/                      # Benchmarking & evaluation
│
├── faiss_index/                      # Generated FAISS vector database
│
├── src/
│   ├── assets/                       # Rewuried png files
│   ├── app.py                        # Streamlit application
│   ├── chatbot.py                    # Chat interface
│   ├── rag.py                        # Main RAG pipeline
│   └── build_index.py                # Knowledge base indexing
│
├── .env                              # Environment variables
├── .gitignore
├── README.md
└── requirements.txt
```
---

# 📸 Demo
### Main Interface

The SRM Academic RAG Assistant provides a clean conversational interface for interacting with official university documents. Users can upload PDFs, build the knowledge base, ask natural language questions, and view the exact source documents and page numbers used to generate each response.

<p align="center">
  <img src="src/assets/home.png" alt="SRM Academic RAG Assistant" width="900"/>
</p>

---

# 🚀 Installation

## Prerequisites

Before running the project, ensure the following are installed:

- Python **3.14.3** (tested)
- Git
- A Groq API Key

---

### 1. Clone the Repository

```bash
git clone https://github.com/gr-hemanth/srm-academic-rag-assistant.git

cd srm-academic-rag-assistant
```

### 2. Create a Virtual Environment

```bash
python -m venv .venv
```

Activate the environment
**Windows (PowerShell)**

```powershell
.\.venv\Scripts\Activate.ps1
```

**Windows (Command Prompt)**

```cmd
.venv\Scripts\activate.bat
```

**Linux / macOS**

```bash
source .venv/bin/activate
```
---

### 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

### 4. Configure Environment Variables

Create a `.env` file in the project root.

```env
GROQ_API_KEY=your_groq_api_key
```

---

### 5. Build the Knowledge Base

Before launching the application for the first time, generate the FAISS vector index from the PDF documents in the `data/` directory.

```bash
python src/build_index.py
```

This command processes the documents, generates embeddings, and creates the FAISS index used for retrieval.

---

### 6. Run the Application

```bash
streamlit run src/app.py
```

The application will be available at:

```
http://localhost:8501
```

---

# 💡 Usage

### Building the Knowledge Base

1. Launch the application.
2. Upload one or more PDF documents.
3. Click **Build Index**.
4. Wait for indexing to complete.
5. Start asking questions.

---

# 🔮 Future Roadmap
Version 2.0

- Docling integration for advanced document parsing
- Metadata-aware retrieval
- Improved retrieval strategies
- Docker support
- Cloud deployment
- Extended benchmark suite
---

# 👨‍💻 Author

**G R Hemanth**

Computer Science Engineering Student

If you found this project helpful, consider giving it a ⭐ on GitHub.

---
## 🤝 Contributing

Contributions, ideas, and feedback are always welcome.

Feel free to **fork** this repository, open an issue, or submit a pull request if you'd like to improve the project.

Whether it's fixing a bug, improving the documentation, or adding a new feature, every contribution is appreciated.

---
