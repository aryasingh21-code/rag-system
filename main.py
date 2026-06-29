import os
import yaml
import requests
from dotenv import load_dotenv
import re
from collections import defaultdict
from fastapi import FastAPI
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from fastapi import HTTPException
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder
import logging

from contextlib import asynccontextmanager
# -------------------------------------------------
# Load environment
# -------------------------------------------------

load_dotenv()

# When running in Docker, docker-compose sets OLLAMA_HOST to the ollama
# service name (http://ollama:11434). Falls back to localhost for local dev.
# AFTER
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")
CHROMA_DB_PATH = os.environ.get("CHROMA_DB_PATH", "./chroma_db")
logging.basicConfig(

    level=logging.INFO,

    format="%(asctime)s | %(levelname)s | %(message)s"

)

logger = logging.getLogger(__name__)


def tokenize(text):
    return re.findall(r"\w+", text.lower())
# -------------------------------------------------
# Caches
# -------------------------------------------------

retrieval_cache: dict[str, list] = {}
answer_cache: dict[str, dict] = {}

# -------------------------------------------------
# FastAPI
# -------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):

    global embedding_model
    global vectordb
    global bm25
    global reranker
    global all_texts
    global all_metadatas

    logger.info("Loading embedding model...")

    embedding_model = HuggingFaceEmbeddings(
        model_name="BAAI/bge-base-en-v1.5",
        encode_kwargs={
            "normalize_embeddings": True
        }
    )

    logger.info("Loading Chroma database...")

    vectordb = Chroma(
        persist_directory=CHROMA_DB_PATH,
        embedding_function=embedding_model,
    )

    all_docs = vectordb.get()

    all_texts = all_docs["documents"]
    if not all_texts:
        raise RuntimeError(
            "Chroma database is empty. Run ingest.py before starting the API."
        )

    all_metadatas = all_docs["metadatas"]

    logger.info("Building BM25 index...")

    tokenized_corpus = [
        tokenize(doc)
        for doc in all_texts
    ]

    bm25 = BM25Okapi(tokenized_corpus)

    logger.info("Loading reranker...")

    reranker = CrossEncoder(
        "BAAI/bge-reranker-base"
    )

    logger.info("Application Ready")

    yield

    logger.info("Application shutting down")

app = FastAPI(
    title="Production RAG API",
    lifespan=lifespan
)

# -------------------------------------------------
# Load Prompt Config
# -------------------------------------------------

with open("prompts/retrieval_v1.yaml", "r") as f:
    prompt_config = yaml.safe_load(f)

PROMPT_TEMPLATE = prompt_config["user_prompt_template"]

MODEL = prompt_config["parameters"]["model"]

TOP_K = prompt_config["parameters"]["top_k_retrieval"]

TOP_N = prompt_config["parameters"]["top_n_rerank"]

USE_MULTI_QUERY = prompt_config["parameters"]["use_multi_query"]

QUERY_VARIANTS = prompt_config["parameters"]["query_variants"]

DECLINE_MSG = prompt_config["citation"]["decline_message"]

SUPPORT_THRESHOLD = prompt_config["citation"]["support_threshold"]


# -------------------------------------------------
# API Models
# -------------------------------------------------

class QuestionRequest(BaseModel):
    question: str



class SourceChunk(BaseModel):
    text: str
    source: str
    page: int | None
    paragraph: int | None
    chunk_id: str | None       # unique ID assigned at ingestion time, if present
    rerank_score: float | None # cross-encoder confidence (higher = more relevant)
    rrf_rank: int | None       # position in the RRF-fused list before reranking (1-based)
class AnswerResponse(BaseModel):
    question: str
    answer: str
    sources: list[SourceChunk]
    cached: bool = False


# -------------------------------------------------
# Hybrid Search  (with retrieval cache)
# -------------------------------------------------

def hybrid_search(question, top_k=20):

    # -------------------------------
    # Retrieval Cache Check
    # -------------------------------

    cache_key = f"{question}||{top_k}"

    if cache_key in retrieval_cache:
        logger.info(f"[retrieval_cache] HIT for: {question!r}")
        return retrieval_cache[cache_key]

    logger.info(f"[retrieval_cache] MISS for: {question!r} — running full retrieval")

    # -------------------------------
    # Dense Retrieval (Chroma)
    # -------------------------------

    # Use similarity_search (no score needed — RRF only uses rank)
    vector_results = vectordb.similarity_search(
        question,
        k=top_k
    )

    # -------------------------------
    # BM25 Retrieval
    # -------------------------------

    bm25_scores = bm25.get_scores(tokenize(question))

    bm25_ranked = sorted(
        enumerate(bm25_scores),
        key=lambda x: x[1],
        reverse=True
    )

    # -------------------------------
    # Reciprocal Rank Fusion (RRF)
    # -------------------------------
    # Formula: RRF(d) = sum of 1 / (k + rank) across retrievers
    # k=60 is the standard constant that dampens the top-rank advantage
    # and prevents one retriever dominating when the other disagrees.

    RRF_K = 60

    rrf_scores: dict[str, float] = defaultdict(float)

    # Track metadata per chunk text so we can return it after fusion
    chunk_metadata: dict[str, dict] = {}

    # Dense ranks (rank 0 = best match)
    for rank, doc in enumerate(vector_results):
        text = doc.page_content
        rrf_scores[text] += 1 / (RRF_K + rank + 1)
        chunk_metadata[text] = doc.metadata

    # BM25 ranks (only top_k considered, same budget as dense)
    for rank, (idx, _) in enumerate(bm25_ranked[:top_k]):
        text = all_texts[idx]
        rrf_scores[text] += 1 / (RRF_K + rank + 1)
        if text not in chunk_metadata:
            chunk_metadata[text] = all_metadatas[idx]

    # -------------------------------
    # Sort by RRF score
    # -------------------------------

    ranked = sorted(
        rrf_scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

    result = [
        {"text": text, "metadata": chunk_metadata[text]}
        for text, _ in ranked[:top_k]
    ]

    # Store in retrieval cache before returning
    retrieval_cache[cache_key] = result

    return result

# -------------------------------------------------
# Reranking
# -------------------------------------------------

def rerank(question, chunks, top_n=5):
    """
    Returns top_n chunks enriched with two extra keys:
      - rerank_score : raw cross-encoder logit (higher = more relevant)
      - rrf_rank     : 1-based position in the RRF candidate list (before reranking)
    """

    pairs = [
        [question, c["text"]]
        for c in chunks
    ]

    scores = reranker.predict(pairs)

    # Attach rrf_rank (position in the incoming RRF-sorted list) before sorting
    ranked = sorted(
        zip(scores, enumerate(chunks, start=1)),
        key=lambda x: x[0],
        reverse=True
    )

    results = []
    for score, (rrf_rank, chunk) in ranked[:top_n]:
        enriched = dict(chunk)
        enriched["rerank_score"] = round(float(score), 4)
        enriched["rrf_rank"] = rrf_rank
        results.append(enriched)

    return results

# -------------------------------------------------
# Query Expansion (Multi-Query Retrieval)
# -------------------------------------------------
 
def expand_query(question, n_variants=2):
    """
    Asks the LLM to generate alternate phrasings of the question to
    improve lexical/semantic overlap with source text during retrieval.
    Returns a list including the original question plus n_variants
    rewrites. Falls back to just the original question if the LLM call
    fails or returns something unusable.
    """
 
    prompt = (
        f"Rewrite the following question in {n_variants} different ways "
        f"that preserve its meaning but use different wording or keywords. "
        f"Output ONLY the rewrites, one per line, with no numbering, "
        f"no extra commentary.\n\n"
        f"Question: {question}"
    )
 
    try:
        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )
        raw = response.json()["response"]
 
        variants = [
            line.strip()
            for line in raw.split("\n")
            if line.strip()
        ][:n_variants]
 
    except Exception as e:
        print(f"[expand_query] LLM call failed, falling back to original only: {e}")
        variants = []
 
    all_queries = [question] + variants
 
    print(f"\n[expand_query] Generated {len(variants)} variant(s):")
    for v in variants:
        print(f"  - {v}")
 
    return all_queries
 
 
def multi_query_search(question, top_k=20, n_variants=2):
    """
    Runs hybrid_search for the original question plus n_variants
    LLM-generated rewrites, then merges and deduplicates the results
    by chunk text, keeping the highest score seen for each chunk.
    Note: each individual hybrid_search call benefits from retrieval
    caching, so repeated variant queries are served from cache.
    """
 
    queries = expand_query(question, n_variants=n_variants)
 
    merged = {}
 
    for q in queries:
        results = hybrid_search(q, top_k)
 
        for r in results:
            text = r["text"]
            if text not in merged:
                merged[text] = r
 
    merged_list = list(merged.values())
 
    print(f"[multi_query_search] {len(queries)} queries -> "
          f"{len(merged_list)} unique candidates (before rerank)")
 
    return merged_list
 
 
# -------------------------------------------------
# Citation Enforcement
# -------------------------------------------------

def is_supported(answer, top_chunks):
    if not top_chunks:
        return False

    # Cross-encoder expects (query, passage) — use answer as query
    # but better: check if ANY chunk supports the answer semantically
    pairs = [[c["text"], answer] for c in top_chunks]  # (passage, claim)
    scores = reranker.predict(pairs)

    best = max(scores)
    logger.info(f"[is_supported] best score: {best:.4f}, threshold: {SUPPORT_THRESHOLD}")
    return best >= SUPPORT_THRESHOLD
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


@app.delete("/cache")
def clear_cache():
    """Clear both retrieval and answer caches (useful after re-ingestion)."""
    retrieval_cache.clear()
    answer_cache.clear()
    logger.info("Both caches cleared via /cache endpoint")
    return {"message": "Retrieval and answer caches cleared."}


# -------------------------------------------------
# Ask  (with answer cache)
# -------------------------------------------------
 

@app.post("/ask", response_model=AnswerResponse)
def ask(request: QuestionRequest):

    # -------------------------------
    # Answer Cache Check
    # -------------------------------

    question_key = request.question.strip().lower()

    if question_key in answer_cache:
        logger.info(f"[answer_cache] HIT for: {request.question!r}")
        cached = answer_cache[question_key]
        return AnswerResponse(
            question=request.question,
            answer=cached["answer"],
            sources=cached["sources"],
            cached=True,
        )

    logger.info(f"[answer_cache] MISS for: {request.question!r} — running full pipeline")

    # -------------------------------
    # Retrieval
    # -------------------------------

    if USE_MULTI_QUERY:
        candidates = multi_query_search(
            request.question,
            TOP_K,
            QUERY_VARIANTS
        )
    else:
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
    logger.info(f"Question: {request.question}")
 
    logger.info("Top retrieved chunks")
 
    for i, chunk in enumerate(top_chunks, 1):
        print(f"\n----- Chunk {i} -----")
        print(chunk["text"][:800])
 
    context = "\n\n".join(c["text"] for c in top_chunks)
 
    prompt = PROMPT_TEMPLATE.format(
        context=context,
        question=request.question
    )
 
    try:

        response = requests.post(
            f"{OLLAMA_HOST}/api/generate",
            json={
                "model": MODEL,
                "prompt": prompt,
                "stream": False
            },
            timeout=180
        )

        response.raise_for_status()

        answer = response.json()["response"]

    except requests.exceptions.Timeout:

        raise HTTPException(
            status_code=504,
            detail="LLM request timed out."
        )

    except requests.exceptions.ConnectionError:

        raise HTTPException(
            status_code=503,
            detail="Unable to connect to Ollama."
        )

    except Exception as e:
        logger.exception(e)
        raise HTTPException(
            status_code=500,
            detail=f"Generation failed: {str(e)}"
        )
 
    # -------------------------------------------------
    # Citation Enforcement
    # -------------------------------------------------
 
    if answer.strip() == DECLINE_MSG:
        sources = []
 
    elif not is_supported(answer, top_chunks):
        answer = DECLINE_MSG
        sources = []
 
    else:
        sources = []
        seen = set()

        for chunk in top_chunks:

            meta = chunk["metadata"]

            raw_page = meta.get("page")
            page = raw_page + 1 if raw_page is not None else None

            paragraph = meta.get("paragraph")

            source_file = meta.get("source", "unknown")
            if not source_file.startswith("http"):
                source_file = os.path.basename(source_file)

            # chunk_id: use whatever key your ingestion pipeline stored
            # (common names: "chunk_id", "id", "doc_id" — falls back to None)
            chunk_id = (
                meta.get("chunk_id")
                or meta.get("id")
                or meta.get("doc_id")
            )

            key = (source_file, page, paragraph)

            if key not in seen:
                seen.add(key)
                sources.append({
                    "text": chunk["text"],
                    "source": source_file,
                    "page": page,
                    "paragraph": paragraph,
                    "chunk_id": str(chunk_id) if chunk_id is not None else None,
                    "rerank_score": chunk.get("rerank_score"),
                    "rrf_rank": chunk.get("rrf_rank"),
                })
 
    logger.info(f"Answer: {answer}")
    logger.info(f"Supported: {answer != DECLINE_MSG}")
    print("=" * 80)

    # -------------------------------
    # Store in Answer Cache
    # -------------------------------

    # Only cache successful, supported answers — not declines
    if answer != DECLINE_MSG:
        answer_cache[question_key] = {
            "answer": answer,
            "sources": sources,
        }
        logger.info(f"[answer_cache] Stored answer for: {request.question!r}")
 
    return AnswerResponse(
        question=request.question,
        answer=answer,
        sources=sources,
        cached=False,
    )