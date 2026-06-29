"""
debug_retrieval.py

Standalone diagnostic for a specific failing query. Imports your real
multi_query_search/rerank functions from main.py so it sees exactly
what your live system sees, but skips the final LLM answer call so you
can inspect retrieval in isolation.

Usage:
    python debug_retrieval.py
"""

from main import multi_query_search, hybrid_search, rerank, TOP_K, TOP_N, all_texts, all_metadatas
import os

QUESTION = "What hardware was used to train the Transformer?"


def show_chunk(label, text, metadata=None):
    print(f"\n--- {label} ---")
    if metadata:
        page = metadata.get("page", "?")
        source = os.path.basename(metadata.get("source", "?"))
        print(f"[{source} | page {page}]")
    print(text[:500])


def main():
    print(f"QUESTION: {QUESTION}\n")

    # Step 1: plain hybrid search on the original question only (old behavior)
    # kept here for comparison so you can see what multi-query adds on top
    plain_candidates = hybrid_search(QUESTION, TOP_K)
    print(f"[baseline] plain hybrid_search returned {len(plain_candidates)} candidates "
          f"(single query, TOP_K={TOP_K})")

    # Step 2: multi-query search (original question + LLM-generated rewrites)
    candidates = multi_query_search(QUESTION, top_k=TOP_K, n_variants=2)
    print(f"\n[multi-query] returned {len(candidates)} unique merged candidates\n")

    for i, c in enumerate(candidates, 1):
        show_chunk(f"Candidate {i}", c["text"], c["metadata"])

    # Step 3: after cross-encoder reranking
    top_chunks = rerank(QUESTION, candidates, TOP_N)
    print(f"\n\n=== AFTER RERANK (top {TOP_N}) ===")
    for i, c in enumerate(top_chunks, 1):
        show_chunk(f"Reranked #{i}", c["text"], c["metadata"])

    # Step 4: did the known-correct chunk make it into the final top N?
    target_phrase = "8 nvidia p100 gpus"
    found_in_final = any(
        target_phrase in c["text"].lower()
        for c in top_chunks
    )

    print(f"\n\n=== RESULT ===")
    if found_in_final:
        print(f"✅ FIXED: a chunk containing '{target_phrase}' is in the final top {TOP_N}.")
    else:
        print(f"❌ STILL MISSING: no chunk containing '{target_phrase}' survived to the final top {TOP_N}.")
        print("   Consider: increasing TOP_N, increasing n_variants in multi_query_search,")
        print("   or trying a different cross-encoder model.")

    # Step 5: full corpus sanity check (unchanged from before)
    print("\n\n=== SEARCHING FULL CORPUS FOR 'P100' / 'GPU' (sanity check) ===")
    found_any = False
    for i, text in enumerate(all_texts):
        if "p100" in text.lower() or "gpu" in text.lower():
            found_any = True
    print(f"Corpus contains {'at least one' if found_any else 'NO'} chunk mentioning P100/GPU.")


if __name__ == "__main__":
    main()