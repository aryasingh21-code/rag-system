"""
calibrate_threshold.py
 
Runs known-good questions (from golden_dataset.json) and known-bad
questions (off-topic, should be declined) through /ask, and prints the
cross-encoder support score for each so you can pick a threshold that
separates the two groups.
 
Usage:
    python calibrate_threshold.py
"""
 
import json
import requests
from sentence_transformers import CrossEncoder
 
API_URL = "http://127.0.0.1:8000/ask"
 
# Same reranker model used in main.py, so scores are directly comparable
reranker = CrossEncoder("BAAI/bge-reranker-base")
 
# Off-topic questions that should NOT be answerable from your PDFs
# (Transformer paper + BERT paper). Edit these if your documents differ.
OFF_TOPIC_QUESTIONS = [
    "What is the capital of France?",
    "How do I bake a chocolate cake?",
    "Who won the cricket world cup in 2011?",
    "What is the boiling point of water in Fahrenheit?",
    "Recommend a good sci-fi movie to watch tonight.",
]
 
 
def get_support_score(answer, sources):
    """Re-derive the cross-encoder max score for an answer vs its sources."""
    if not sources:
        return None
    pairs = [[answer, s["text"]] for s in sources]
    scores = reranker.predict(pairs)
    return float(max(scores))
 
 
def run_batch(questions, label):
    print(f"\n{'=' * 70}")
    print(f"  {label}  ({len(questions)} questions)")
    print(f"{'=' * 70}")
 
    results = []
 
    for q in questions:
        res = requests.post(API_URL, json={"question": q})
        body = res.json()
        answer = body["answer"]
        sources = body.get("sources", [])
 
        score = get_support_score(answer, sources)
 
        results.append({
            "question": q,
            "answer_preview": answer[:80],
            "support_score": round(score, 3) if score is not None else None,
        })
 
        score_str = f"{score:.3f}" if score is not None else "N/A (no sources)"
        print(f"\nQ: {q}")
        print(f"  Support Score: {score_str}")
        print(f"  Answer: {answer[:100]}")
 
    return results
 
 
def main():
    with open("golden_dataset.json", "r", encoding="utf-8") as f:
        golden = json.load(f)["pairs"]
 
    good_questions = [item["question"] for item in golden]
 
    good_results = run_batch(good_questions, "KNOWN-GOOD QUESTIONS (should be answered)")
    bad_results = run_batch(OFF_TOPIC_QUESTIONS, "KNOWN-BAD QUESTIONS (should be declined)")
 
    good_scores = [r["support_score"] for r in good_results if r["support_score"] is not None]
    bad_scores = [r["support_score"] for r in bad_results if r["support_score"] is not None]
 
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
 
    if good_scores:
        print(f"Good questions  -> min: {min(good_scores):.3f}  max: {max(good_scores):.3f}  "
              f"avg: {sum(good_scores)/len(good_scores):.3f}")
    if bad_scores:
        print(f"Bad questions   -> min: {min(bad_scores):.3f}  max: {max(bad_scores):.3f}  "
              f"avg: {sum(bad_scores)/len(bad_scores):.3f}")
 
    if good_scores and bad_scores:
        suggested = (min(good_scores) + max(bad_scores)) / 2
        print(f"\nSuggested threshold (midpoint between worst-good and best-bad): {suggested:.3f}")
        if min(good_scores) <= max(bad_scores):
            print("⚠️  WARNING: good and bad score ranges overlap. No single threshold")
            print("   will perfectly separate them — inspect the per-question scores")
            print("   above and consider whether 'bad' questions are returning sources")
            print("   at all (if hybrid_search has no relevance floor, BM25 may still")
            print("   return something for any query).")
 
    with open("calibration_results.json", "w") as f:
        json.dump({
            "good_questions": good_results,
            "bad_questions": bad_results,
        }, f, indent=2)
    print("\nFull results saved to calibration_results.json")
 
 
if __name__ == "__main__":
    main()