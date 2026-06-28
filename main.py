import os
import yaml
import requests
from dotenv import load_dotenv
import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from google import genai

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

# -------------------------------------------------
# Load environment
# -------------------------------------------------

load_dotenv()

API_KEY = os.getenv("GEMINI_API_KEY")

gemini_client = genai.Client(api_key=API_KEY)

# -------------------------------------------------
# FastAPI
# -------------------------------------------------

app = FastAPI(title="Production RAG API")

# -------------------------------------------------
# Load Prompt Config
# -------------------------------------------------

with open("prompts/retrieval_v1.yaml", "r") as f:
    prompt_config = yaml.safe_load(f)

PROMPT_TEMPLATE = prompt_config["user_prompt_template"]

MODEL = prompt_config["parameters"]["model"]

TOP_K = prompt_config["parameters"]["top_k_retrieval"]

TOP_N = prompt_config["parameters"]["top_n_rerank"]

# -------------------------------------------------
# Local Embeddings
# -------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={
        "normalize_embeddings": True
    }
)

# -------------------------------------------------
# Load Chroma
# -------------------------------------------------



vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding_model,
)

all_docs = vectordb.get()

all_texts = all_docs["documents"]

all_metadatas = all_docs["metadatas"]







# -------------------------------------------------
# BM25
# -------------------------------------------------

def tokenize(text):
    return re.findall(r"\w+", text.lower())

tokenized_corpus = [
    tokenize(doc)
    for doc in all_texts
]

bm25 = BM25Okapi(tokenized_corpus)

# -------------------------------------------------
# Cross Encoder
# -------------------------------------------------

reranker = CrossEncoder(
    "BAAI/bge-reranker-base"
)

# -------------------------------------------------
# API Models
# -------------------------------------------------

class QuestionRequest(BaseModel):
    question: str


class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[str]


# -------------------------------------------------
# Hybrid Search
# -------------------------------------------------

def hybrid_search(question, top_k=20):

    # -------------------------------
    # Dense Retrieval (Chroma)
    # -------------------------------

    vector_results = vectordb.similarity_search_with_score(
        question,
        k=top_k
    )

    # -------------------------------
    # BM25 Retrieval
    # -------------------------------

    bm25_scores = bm25.get_scores(tokenize(question))

    # -------------------------------
    # Combine Scores
    # -------------------------------

    scores = {}

    # Dense similarity
    # Chroma returns distance (smaller is better)
    for doc, distance in vector_results:

        text = doc.page_content

        dense_score = 1 / (1 + distance)

        scores[text] = {
            "score": dense_score,
            "metadata": doc.metadata
        }

    # BM25
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1

    for idx, score in enumerate(bm25_scores):

        text = all_texts[idx]

        normalized_bm25 = score / max_bm25

        if text in scores:

            scores[text]["score"] += normalized_bm25

        else:

            scores[text] = {
                "score": normalized_bm25,
                "metadata": all_metadatas[idx]
            }

    # -------------------------------
    # Sort
    # -------------------------------

    ranked = sorted(
        scores.items(),
        key=lambda x: x[1]["score"],
        reverse=True
    )

    return [
        text
        for text, _ in ranked[:top_k]
    ]
# -------------------------------------------------
# Reranking
# -------------------------------------------------

def rerank(question, chunks, top_n=5):

    pairs = [
        [question, chunk]
        for chunk in chunks
    ]

    scores = reranker.predict(pairs)

    ranked = sorted(
        zip(scores, chunks),
        reverse=True
    )

    return [
        chunk
        for _, chunk in ranked[:top_n]
    ]

# -------------------------------------------------
# Metadata
# -------------------------------------------------

def get_metadata(chunk):

    for i, text in enumerate(all_texts):

        if text == chunk:

            return all_metadatas[i]

    return {}

# -------------------------------------------------
# Routes
# -------------------------------------------------

@app.get("/")
def home():

    return {
        "message": "Production RAG API Running"
    }


@app.get("/ui", response_class=HTMLResponse)
def ui():

    with open(
        "templates/index.html",
        encoding="utf-8"
    ) as f:

        return f.read()

# -------------------------------------------------
# Ask
# -------------------------------------------------

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):

    candidates = hybrid_search(
        request.question,
        TOP_K
    )

    top_chunks = rerank(
        request.question,
        candidates,
        TOP_N
    )

    print("\n" + "=" * 80)
    print("QUESTION:")
    print(request.question)

    print("\nTOP CHUNKS:")

    for i, chunk in enumerate(top_chunks, 1):
        print(f"\n----- Chunk {i} -----")
        print(chunk[:800])

    context = "\n\n".join(top_chunks)

    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=request.question
    )

    response = requests.post(
        "http://localhost:11434/api/generate",
        json={
            "model": "llama3.2",
            "prompt": prompt,
            "stream": False
        }
    )

    answer = response.json()["response"]

    sources = []

    for chunk in top_chunks:

        meta = get_metadata(chunk)

        if meta:

            source = (
                f"Page {meta.get('page',0)+1} "
                f"of {os.path.basename(meta.get('source','document.pdf'))}"
            )

            if source not in sources:
                sources.append(source)

    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=sources,
    )