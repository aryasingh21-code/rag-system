from langchain_chroma import Chroma
from langchain_core.embeddings import Embeddings
from google import genai
from typing import List

# Same Gemini Embeddings wrapper
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

from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

# Load existing ChromaDB
embeddings = GeminiEmbeddings(api_key=API_KEY)
vectordb = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embeddings
)

# User question
question = "What are the limitations of recurrent models?"
# Retrieve top 5 relevant chunks
retriever = vectordb.as_retriever(search_kwargs={"k": 5})
relevant_chunks = retriever.invoke(question)

# Build context from chunks
context = "\n\n".join([chunk.page_content for chunk in relevant_chunks])

# Build prompt
prompt = f"""Answer the question using ONLY the context below.
If the answer is not in the context, say "I don't have enough information."

Context:
{context}

Question:
{question}
"""

# Generate answer using Gemini
client = genai.Client(api_key=API_KEY)
response = client.models.generate_content(
    model="models/gemini-2.5-flash",
    contents=prompt
)

print("=== ANSWER ===")
print(response.text)

print("\n=== SOURCES ===")
for chunk in relevant_chunks:
    print(f"- Page {chunk.metadata['page'] + 1} of {chunk.metadata['source']}")