from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from google import genai
from langchain_core.embeddings import Embeddings
from typing import List

# Custom Gemini Embeddings wrapper
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

# Load and chunk
loader = PyPDFLoader("document.pdf")
docs = loader.load()
splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=100)
chunks = splitter.split_documents(docs)

# Store in ChromaDB
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")
embeddings = GeminiEmbeddings(api_key=API_KEY)
vectordb = Chroma.from_documents(
    chunks,
    embeddings,
    persist_directory="./chroma_db"
)

print(f"✅ Stored {len(chunks)} chunks in ChromaDB!")