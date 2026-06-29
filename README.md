# 📄 RAG Document Assistant

> Ask natural-language questions against your own PDFs and websites — powered by Hybrid Search, Cross-Encoder Reranking, and a local Ollama LLM.

---

## How it works

```
Question → Hybrid Search (Dense + BM25 + RRF) → Cross-Encoder Rerank → Ollama LLM → Citation Check → Answer
```

1. **Hybrid retrieval** — ChromaDB (dense) + BM25 keyword search, fused with Reciprocal Rank Fusion
2. **Reranking** — `BAAI/bge-reranker-base` cross-encoder scores and re-orders candidates
3. **Generation** — `llama3.2` via Ollama produces an answer from the top-N chunks
4. **Citation enforcement** — answer is cross-encoder scored against chunks; declined if not grounded

---

## Stack

| Layer | Tech |
|---|---|
| Embedding | `BAAI/bge-base-en-v1.5` |
| Vector store | ChromaDB |
| Keyword search | BM25Okapi |
| Fusion | Reciprocal Rank Fusion (k=60) |
| Reranker | `BAAI/bge-reranker-base` |
| LLM | `llama3.2` via Ollama |
| API | FastAPI + Uvicorn |
| UI | Vanilla HTML/JS at `/ui` |
| Infra | Docker Compose |

---

## Project structure

```
rag-project/
├── main.py                    # FastAPI app — search, rerank, generate, cache
├── ingest.py                  # Load PDFs + websites → chunk → embed → ChromaDB
├── evaluate_rag.py            # Evaluation: answer similarity, faithfulness, Hit@K, MRR
├── calibrate_threshold.py     # Find the right support_threshold
├── prompts/
│   └── retrieval_v1.yaml      # All config: model, chunk size, top_k, threshold, prompt
├── templates/
│   └── index.html             # Browser UI
├── documents/                 # Drop your PDFs here
├── websites.txt               # One URL per line
├── chroma_db/                 # Auto-generated (do not edit)
├── golden_dataset.json        # Ground-truth Q&A for evaluation
├── docker-compose.yml
├── Dockerfile
└── requirements.txt
```

---

## Quick start

### Prerequisites
- Docker Desktop running
- 8 GB+ RAM

### 1. Add your documents

Drop PDFs into `documents/`. Add website URLs (one per line) to `websites.txt`.

### 2. Start the stack

```bash
docker compose up -d
```

Three containers start:
- `ollama` — local LLM runtime
- `ingest` — indexes documents, then exits
- `rag-api` — FastAPI backend + UI

### 3. Open the UI

```
http://localhost:8000/ui
```

### 4. Or call the API

```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What does BERT stand for?"}'
```

---

## Configuration

Everything lives in `prompts/retrieval_v1.yaml`:

```yaml
parameters:
  model: "llama3.2"
  top_k_retrieval: 20      # candidates from hybrid search
  top_n_rerank: 8          # kept after reranking
  chunk_size: 700          # tokens per chunk
  chunk_overlap: 100
  use_multi_query: false   # LLM query expansion
  query_variants: 2

citation:
  support_threshold: 0.35  # min cross-encoder score to accept an answer
  decline_message: "I don't have enough information..."
```

---

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/ui` | GET | Browser UI |
| `/ask` | POST | Ask a question |
| `/cache` | DELETE | Clear retrieval + answer caches |

### POST `/ask`

**Request**
```json
{ "question": "What does BERT stand for?" }
```

**Response**
```json
{
  "question": "What does BERT stand for?",
  "answer": "BERT stands for Bidirectional Encoder Representations from Transformers.",
  "cached": false,
  "sources": [
    {
      "text": "...",
      "source": "bert.pdf",
      "page": 1,
      "paragraph": 2,
      "rerank_score": 0.9123,
      "rrf_rank": 1
    }
  ]
}
```

---

## Caching

Two in-memory caches cut latency on repeated queries:

- **Retrieval cache** — skips hybrid search + reranking
- **Answer cache** — skips the entire pipeline including LLM

Only supported answers are cached. Declined answers never are. After re-ingesting documents, clear both:

```bash
curl -X DELETE http://localhost:8000/cache
```

---

## Evaluation

Requires a `golden_dataset.json`:

```json
{
  "pairs": [
    { "question": "...", "answer": "...", "source": "bert.pdf", "page": 4, "paragraph": 2 }
  ]
}
```

```bash
python evaluate_rag.py
```

Metrics reported:

| Metric | Description |
|---|---|
| Answer Similarity | Cosine similarity between predicted and gold answer |
| Faithfulness | How grounded the answer is in retrieved chunks |
| Hit@K | Did a matching chunk appear in top-K? |
| MRR | Mean Reciprocal Rank |
| Latency | End-to-end time per question |

Add `--ci` to fail the process if thresholds aren't met:

```bash
python evaluate_rag.py --ci
# Thresholds: Answer Similarity ≥ 0.75 | Faithfulness ≥ 0.70 | Hit@8 ≥ 0.90
```

---

## Threshold calibration

If valid answers are being declined, the `support_threshold` may be too high. Run:

```bash
python calibrate_threshold.py
```

It scores your golden questions (should pass) vs off-topic questions (should be declined) and suggests a midpoint threshold. Results saved to `calibration_results.json`.

---

## Troubleshooting

| Symptom | Fix |
|---|---|
| UI says "don't have enough information" but API works | `support_threshold` too high — run `calibrate_threshold.py` |
| UI shows old version after editing `index.html` | `docker cp templates/index.html rag-api:/app/templates/index.html` then `Ctrl+Shift+R` |
| `Chroma DB is empty` on startup | Run ingest first: `docker compose run ingest` |
| Ollama timeout (504) | Model still loading or prompt too long — reduce `top_n_rerank` |
| Stale cached answer | `curl -X DELETE http://localhost:8000/cache` |

---

## Known limitations

- Caches are in-memory — reset on container restart (no Redis persistence)
- No authentication on API or UI
- No document upload UI — adding docs requires shell access + rebuild
- LLM responses are not streamed — full answer waits before returning
- Each question is stateless — no conversation memory
- `chunk_id` is always `null` — ingest doesn't assign stable IDs