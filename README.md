# Production RAG System

A production-grade Retrieval Augmented Generation (RAG) system 
that answers questions about documents with citations.

## Architecture
Documents → Chunking → Embeddings → ChromaDB → Retrieval → Gemini → Answer + Citations

## Tech Stack
- FastAPI
- ChromaDB
- Google Gemini (Embeddings + LLM)
- LangChain

## Setup
1. Clone the repo
2. Install dependencies: `pip install -r requirements.txt`
3. Add your Gemini API key to `.env`
4. Run: `python store.py`
5. Start API: `uvicorn main:app --reload`

## API Usage
POST /ask
{"question": "What is the attention mechanism?"}

## Live Demo
https://rag-system-production-7300.up.railway.app/ui