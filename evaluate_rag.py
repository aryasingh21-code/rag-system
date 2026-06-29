import json
import sys
import time
import requests
from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer("BAAI/bge-base-en-v1.5")
import os

API_URL = os.getenv(
    "API_URL",
    "http://127.0.0.1:8000/ask"
)


def semantic_score(pred, gold):
    emb1 = model.encode(pred, convert_to_tensor=True)
    emb2 = model.encode(gold, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))


def faithfulness_score(answer, retrieved_chunks):
    """How well-grounded is the answer in the retrieved context (not the gold answer)."""
    if not retrieved_chunks:
        return 0.0
    context = " ".join(retrieved_chunks)
    return semantic_score(answer, context)


def _matches(retrieved_chunk, gold_source, gold_page, gold_paragraph):
    """
    A retrieved chunk matches the gold label if source and page agree.
    Paragraph is only checked when the gold dataset actually specifies
    one — most entries leave it null until real chunk-level paragraph
    numbers have been verified against the live vector store.
    """
    if retrieved_chunk.get("source") != gold_source:
        return False
    if retrieved_chunk.get("page") != gold_page:
        return False
    if gold_paragraph is not None:
        if retrieved_chunk.get("paragraph") != gold_paragraph:
            return False
    return True


def retrieval_hit(retrieved_sources, gold_source, gold_page, gold_paragraph, k):
    """Did a matching chunk appear in the top-k retrieved chunks?"""
    top_k = retrieved_sources[:k]
    return 1.0 if any(
        _matches(r, gold_source, gold_page, gold_paragraph) for r in top_k
    ) else 0.0


def retrieval_mrr(retrieved_sources, gold_source, gold_page, gold_paragraph):
    """Reciprocal rank of the first matching chunk in the retrieved list."""
    for i, r in enumerate(retrieved_sources):
        if _matches(r, gold_source, gold_page, gold_paragraph):
            return 1.0 / (i + 1)
    return 0.0


def evaluate(k=8):
    with open("golden_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)["pairs"]

    results = []
    answer_sim_scores = []
    faithfulness_scores = []
    hit_at_k_scores = []
    mrr_scores = []
    latencies = []

    print("\n📊 RAG EVALUATION STARTING...\n")

    for item in data:
        question = item["question"]
        expected = item["answer"]
        gold_source = item.get("source")
        gold_page = item.get("page")
        gold_paragraph = item.get("paragraph")

        start = time.time()
        res = requests.post(API_URL, json={"question": question})
        latency = time.time() - start
        latencies.append(latency)

        
        if res.status_code != 200:
            print(f"Request failed: {res.status_code}")
            print(res.text)
            continue

        body = res.json()
        pred = body["answer"]

        retrieved_sources = body.get("sources", [])  # list of {"text","source","page","paragraph"}
        retrieved_chunks = [s["text"] for s in retrieved_sources]

        ans_score = semantic_score(pred, expected)
        faith_score = faithfulness_score(pred, retrieved_chunks)

        # gold_source being present is the minimum requirement to evaluate
        # retrieval (a question always has a source; page/paragraph may be None)
        hit = retrieval_hit(retrieved_sources, gold_source, gold_page, gold_paragraph, k) \
            if gold_source is not None else None
        mrr = retrieval_mrr(retrieved_sources, gold_source, gold_page, gold_paragraph) \
            if gold_source is not None else None

        answer_sim_scores.append(ans_score)
        faithfulness_scores.append(faith_score)
        if hit is not None:
            hit_at_k_scores.append(hit)
            mrr_scores.append(mrr)

        results.append({
            "question": question,
            "answer_similarity": round(ans_score, 3),
            "faithfulness": round(faith_score, 3),
            "hit_at_k": hit,
            "mrr": round(mrr, 3) if mrr is not None else None,
            "latency_sec": round(latency, 2),
        })

        print(f"Q: {question}")
        print(f"  Answer Similarity: {ans_score:.3f} | Faithfulness: {faith_score:.3f} | "
              f"Hit@{k}: {hit} | MRR: {mrr} | Latency: {latency:.2f}s")
        print("-" * 60)

    summary = {
        "avg_answer_similarity": round(sum(answer_sim_scores) / len(answer_sim_scores), 3),
        "avg_faithfulness": round(sum(faithfulness_scores) / len(faithfulness_scores), 3),
        f"hit_at_{k}": round(sum(hit_at_k_scores) / len(hit_at_k_scores), 3) if hit_at_k_scores else None,
        "mrr": round(sum(mrr_scores) / len(mrr_scores), 3) if mrr_scores else None,
        "avg_latency_sec": round(sum(latencies) / len(latencies), 2),
    }

    print("\n🔥 SUMMARY METRICS")
    for k_, v in summary.items():
        print(f"  {k_}: {v}")

    with open("eval_results.json", "w") as f:
        json.dump({"summary": summary, "per_question": results}, f, indent=2)
    print("\nResults saved to eval_results.json")

    if "--ci" in sys.argv:

        if (
            summary["avg_answer_similarity"] < 0.75
            or summary["avg_faithfulness"] < 0.70
            or summary["hit_at_8"] < 0.90
        ):
            print("❌ Quality gate FAILED!")
            sys.exit(1)

        print("✅ Quality gate PASSED!")


if __name__ == "__main__":
    evaluate(k=8)