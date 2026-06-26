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

# Gemini Embeddings wrapper
class GeminiEmbeddings(Embeddings):
    def __init__(self, api_key: str):
        self.client = genai.Client(api_key=api_key)

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        result = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=texts
        )
        return [e.values for e in result.embeddings]

    def embed_query(self, text: str) -> List[float]:
        result = self.client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=text
        )
        return result.embeddings[0].values

# Load ChromaDB
embeddings = GeminiEmbeddings(api_key=API_KEY)
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# Load all chunks for BM25
all_docs = vectordb.get()
all_texts = all_docs["documents"]
all_metadatas = all_docs["metadatas"]

# Build BM25 index
tokenized_corpus = [doc.lower().split() for doc in all_texts]
bm25 = BM25Okapi(tokenized_corpus)

# Load reranker
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

def hybrid_search(question: str, top_k: int = 20) -> list:
    # BM25 search
    tokenized_query = question.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True
    )[:top_k]

    # Vector search
    vector_results = vectordb.similarity_search(question, k=top_k)
    vector_texts = set([doc.page_content for doc in vector_results])

    # Combine results
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

def get_metadata_for_chunk(text: str) -> dict:
    for i, doc_text in enumerate(all_texts):
        if doc_text == text:
            return all_metadatas[i]
    return {}

# --- Main RAG flow ---
question = "How does multi-head attention work?"

print("🔍 Running hybrid search...")
candidates = hybrid_search(question, top_k=20)

print("📊 Reranking...")
top_chunks = rerank(question, candidates, top_n=5)

# Build context
context = "\n\n".join(top_chunks)

# Build prompt
prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question:
{question}
"""

# Generate answer
client = genai.Client(api_key=API_KEY)
response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents=prompt
)

print("\n=== ANSWER ===")
print(response.text)

print("\n=== SOURCES ===")
for chunk in top_chunks:
    meta = get_metadata_for_chunk(chunk)
    if meta:
        print(f"- Page {meta.get('page', 0) + 1} of {meta.get('source', 'document.pdf')}")