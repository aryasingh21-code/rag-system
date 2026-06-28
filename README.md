# 🚀 Production-Grade RAG System

A production-style Retrieval-Augmented Generation (RAG) system that answers questions from PDF documents using **Hybrid Search**, **Cross-Encoder Re-ranking**, and **Llama 3.2**. The system combines semantic retrieval with lexical search to improve retrieval accuracy and provides source citations for every response.

---

## ✨ Features

* 📄 PDF document ingestion
* ✂️ Recursive text chunking
* 🔎 Dense retrieval using **BGE Embeddings**
* 📚 ChromaDB vector database
* 🔤 BM25 lexical retrieval
* ⚡ Hybrid Search (Dense + BM25)
* 🎯 Cross-Encoder Re-ranking
* 🤖 Local LLM inference using **Llama 3.2 (Ollama)**
* 🌐 FastAPI REST API
* 💻 Simple web interface
* 📊 Automated RAG evaluation pipeline
* 📑 Source citations with page numbers

---

## 🏗️ Architecture

```
PDF Documents
      │
      ▼
PyPDFLoader
      │
      ▼
Recursive Text Splitter
      │
      ▼
BGE Embeddings
      │
      ▼
ChromaDB
      │
      ▼
Hybrid Retrieval
(Dense + BM25)
      │
      ▼
Cross-Encoder Re-ranking
      │
      ▼
Prompt Construction
      │
      ▼
Llama 3.2 (Ollama)
      │
      ▼
Answer + Citations
```

---

## 🛠️ Tech Stack

* Python
* FastAPI
* LangChain
* ChromaDB
* Hugging Face Transformers
* BAAI BGE Embeddings
* Sentence Transformers
* BM25
* Cross-Encoder (MS MARCO MiniLM)
* Ollama
* Llama 3.2

---

## 📂 Project Structure

```
production-rag-system/

├── documents/
├── prompts/
├── templates/
├── ingest.py
├── main.py
├── evaluate_rag.py
├── requirements.txt
├── README.md
```

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/aryasingh21-code/rag-system.git
cd rag-system
```

Create a virtual environment

```bash
python -m venv .venv
```

Activate it

Windows

```bash
.venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root.

```
GEMINI_API_KEY=your_api_key
```

> **Note:** The current implementation uses Ollama (Llama 3.2) for answer generation. The Gemini API key is only required if you enable Gemini-based functionality.

---

## 📚 Build the Vector Database

```bash
python ingest.py
```

---

## ▶️ Run the API

```bash
uvicorn main:app --reload
```

Open the web interface:

```
http://localhost:8000/ui
```

---

## 🧪 Evaluate the RAG Pipeline

Run:

```bash
python evaluate_rag.py
```

Current evaluation score:

```
Final RAG Score: 0.827
```

---

## 🔍 Retrieval Pipeline

1. Embed documents using **BAAI/bge-base-en-v1.5**
2. Store embeddings in **ChromaDB**
3. Retrieve candidates using Dense Search
4. Retrieve additional candidates using BM25
5. Merge candidate sets (Hybrid Search)
6. Re-rank using **cross-encoder/ms-marco-MiniLM-L-6-v2**
7. Generate answers using **Llama 3.2**
8. Return answers with document citations

---

## 🤖 Models Used

### Embedding Model

```
BAAI/bge-base-en-v1.5
```

### Re-ranker

```
cross-encoder/ms-marco-MiniLM-L-6-v2
```

### Language Model

```
llama3.2 (Ollama)
```

---

## 📈 Future Improvements

* Reciprocal Rank Fusion (RRF)
* Parent Document Retrieval
* Multi-Query Retrieval
* Metadata Filtering
* Streaming responses
* Docker support
* CI/CD pipeline
* Authentication
* Conversation memory

---

## 👨‍💻 Author

**Arya Singh**

If you found this project useful, feel free to ⭐ the repository.
