"""
evaluate_ragas.py

Runs the standard RAGAS metric suite (faithfulness, answer_relevancy,
context_precision, context_recall) against your RAG system, using your
local Ollama model as the judge LLM instead of OpenAI — so this runs
fully offline with no API key required.

Usage:
    python evaluate_ragas.py
"""

import json
import requests

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings

from ragas import EvaluationDataset, evaluate
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.metrics import (
    Faithfulness,
    AnswerRelevancy,
    ContextPrecision,
    ContextRecall,
)

API_URL = "http://127.0.0.1:8000/ask"
OLLAMA_MODEL = "llama3.2"  # same model your RAG system uses for generation


def build_dataset():
    """
    Calls your live /ask endpoint for every question in golden_dataset.json
    and assembles a RAGAS-compatible dataset: question, retrieved contexts,
    generated answer, and the gold reference answer.
    """

    with open("golden_dataset.json", "r", encoding="utf-8") as f:
        pairs = json.load(f)["pairs"]

    samples = []

    print(f"Querying /ask for {len(pairs)} questions...\n")

    for item in pairs:
        question = item["question"]
        reference = item["answer"]

        res = requests.post(API_URL, json={"question": question})
        body = res.json()

        answer = body["answer"]
        contexts = [s["text"] for s in body.get("sources", [])]

        # RAGAS context-based metrics need at least one retrieved context;
        # skip declined/no-source answers since there's nothing to score
        if not contexts:
            print(f"[skip] no sources returned for: {question}")
            continue

        samples.append({
            "user_input": question,
            "response": answer,
            "retrieved_contexts": contexts,
            "reference": reference,
        })

        print(f"[ok] {question}")

    return EvaluationDataset.from_list(samples)


def main():
    dataset = build_dataset()

    if len(dataset) == 0:
        print("\nNo evaluable samples (every question was declined/had no sources).")
        return

    # Local judge — Ollama instead of OpenAI, so this needs no API key
    judge_llm = LangchainLLMWrapper(ChatOllama(model=OLLAMA_MODEL))

    judge_embeddings = LangchainEmbeddingsWrapper(
        HuggingFaceEmbeddings(model_name="BAAI/bge-base-en-v1.5")
    )

    metrics = [
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision(),
        ContextRecall(),
    ]

    print(f"\nRunning RAGAS evaluation on {len(dataset)} samples "
          f"(judge: {OLLAMA_MODEL} via Ollama)...\n")

    result = evaluate(
        dataset=dataset,
        metrics=metrics,
        llm=judge_llm,
        embeddings=judge_embeddings,
    )

    print("\n" + "=" * 60)
    print("RAGAS RESULTS")
    print("=" * 60)
    print(result)

    df = result.to_pandas()
    df.to_csv("ragas_results.csv", index=False)
    print("\nPer-question results saved to ragas_results.csv")


if __name__ == "__main__":
    main()