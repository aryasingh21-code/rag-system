import json
import requests
from sentence_transformers import SentenceTransformer, util

# Load embedding model (local)

model = SentenceTransformer("BAAI/bge-base-en-v1.5")

API_URL = "http://127.0.0.1:8000/ask"

def semantic_score(pred, gold):
    emb1 = model.encode(pred, convert_to_tensor=True)
    emb2 = model.encode(gold, convert_to_tensor=True)
    return float(util.cos_sim(emb1, emb2))

def evaluate():
    with open("golden_dataset.json", "r", encoding="utf-8") as f:
        data = json.load(f)["pairs"]

    scores = []

    print("\n📊 RAG EVALUATION STARTING...\n")

    for item in data:
        question = item["question"]
        expected = item["answer"]

        # Call your RAG system
        res = requests.post(API_URL, json={"question": question})
        pred = res.json()["answer"]

        score = semantic_score(pred, expected)
        scores.append(score)

        print(f"Q: {question}")
        print(f"Score: {score:.3f}")
        print("-" * 50)

    avg_score = sum(scores) / len(scores)

    print("\n🔥 FINAL RAG SCORE:", round(avg_score, 3))


    # Save results for CI/CD
    import sys
    with open("eval_results.json", "w") as f:
        json.dump({"semantic_similarity": round(avg_score, 3)}, f)
    print("Results saved to eval_results.json")

    # If running in CI mode, exit with error if below threshold
    if "--ci" in sys.argv and avg_score < 0.75:
        print("❌ Quality gate FAILED!")
        sys.exit(1)
    elif "--ci" in sys.argv:
        print("✅ Quality gate PASSED!")

if __name__ == "__main__":
    evaluate()