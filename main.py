from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from google import genai
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
from typing import List
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

app = FastAPI(title="Production RAG API")

import yaml

# Load prompt config
with open("prompts/retrieval_v1.yaml", "r") as f:
    prompt_config = yaml.safe_load(f)

PROMPT_TEMPLATE = prompt_config["user_prompt_template"]
MODEL = prompt_config["parameters"]["model"]
TOP_K = prompt_config["parameters"]["top_k_retrieval"]
TOP_N = prompt_config["parameters"]["top_n_rerank"]

class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        import time
        all_embeddings = []
        batch_size = 20
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            result = self.client.models.embed_content(
                model="models/gemini-embedding-001",
                contents=batch
            )
            all_embeddings.extend([e.values for e in result.embeddings])
            print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} chunks...")
            time.sleep(35)
        return all_embeddings

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter

embeddings = GeminiEmbeddings(api_key=API_KEY)

# Auto-build ChromaDB if empty
chroma_path = "./chroma_db"
vectordb = Chroma(persist_directory=chroma_path, embedding_function=embeddings)
existing = vectordb.get()

if not existing["documents"]:
    print("ChromaDB is empty — building from all PDFs in documents/...")
    all_chunks = []
    pdf_folder = "./documents"
    for filename in os.listdir(pdf_folder):
        if filename.endswith(".pdf"):
            print(f"Loading {filename}...")
            loader = PyPDFLoader(os.path.join(pdf_folder, filename))
            docs = loader.load()
            splitter = RecursiveCharacterTextSplitter(
                chunk_size=700,
                chunk_overlap=100
            )
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)
            print(f"  → {len(chunks)} chunks from {filename}")
    vectordb = Chroma.from_documents(
        all_chunks,
        embeddings,
        persist_directory=chroma_path
    )
    print(f"Done! Total chunks stored: {len(all_chunks)}")

all_docs = vectordb.get()

all_texts = all_docs.get("documents", [])
all_metadatas = all_docs.get("metadatas", [])

if all_texts:
    tokenized_corpus = [doc.lower().split() for doc in all_texts]
    bm25 = BM25Okapi(tokenized_corpus)
else:
    bm25 = None

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
gemini_client = genai.Client(api_key=API_KEY)

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]

def hybrid_search(question: str, top_k: int = 20):
    vector_results = vectordb.similarity_search(question, k=top_k)

    vector_texts = [doc.page_content for doc in vector_results]

    if bm25 is None:
        return vector_texts

    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    combined = list(vector_texts)

    for i in top_bm25_indices:
        if all_texts[i] not in combined:
            combined.append(all_texts[i])

    return combined[:top_k]

def rerank(question: str, chunks: list, top_n: int = 5) -> list:
    pairs = [[question, chunk] for chunk in chunks]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, chunks), reverse=True)
    return [chunk for _, chunk in ranked[:top_n]]

def get_metadata(text: str) -> dict:
    for i, doc_text in enumerate(all_texts):
        if doc_text == text:
            return all_metadatas[i]
    return {}

@app.get("/ui", response_class=HTMLResponse)
def ui():
    with open("templates/index.html", "r", encoding="utf-8") as f:
        return f.read()

@app.get("/")
def root():
    return {"message": "Production RAG API is running!"}

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):
    candidates = hybrid_search(request.question, top_k=TOP_K)
    top_chunks = rerank(request.question, candidates, top_n=TOP_N)
    context = "\n\n".join(top_chunks)
    prompt = PROMPT_TEMPLATE.format(
    context=context,
    question=request.question
)
    response = gemini_client.models.generate_content(
        model="models/gemini-2.5-flash",
        contents=prompt
    )
    sources = []
    for chunk in top_chunks:
        meta = get_metadata(chunk)
        if meta:
            source = f"Page {meta.get('page', 0) + 1} of {meta.get('source', 'document.pdf')}"
            if source not in sources:
                sources.append(source)
    return AnswerResponse(
        question=request.question,
        answer=response.text,
        sources=sources
    )