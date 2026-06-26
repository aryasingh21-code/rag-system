from google import genai
import os
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

client = genai.Client(api_key=API_KEY)

result = client.models.embed_content(
    model="models/gemini-embedding-001",
    contents="Hello world"
)

print(f"Embedding works! Vector length: {len(result.embeddings[0].values)}")