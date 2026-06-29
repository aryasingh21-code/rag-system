import os
import yaml

from transformers import AutoTokenizer

from langchain_community.document_loaders import (
    PyPDFLoader,
    WebBaseLoader,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

import os

os.environ["USER_AGENT"] = "ProductionRAGSystem/1.0"

# -------------------------------------------------
# Configuration
# -------------------------------------------------

PDF_FOLDER = "./documents"
CHROMA_PATH = "./chroma_db"
WEBSITE_FILE = "./websites.txt"
PROMPT_CONFIG_PATH = "./prompts/retrieval_v1.yaml"



# -------------------------------------------------
# Load Chunking Config (single source of truth, shared with main.py)
# -------------------------------------------------

with open(PROMPT_CONFIG_PATH, "r") as f:
    prompt_config = yaml.safe_load(f)

CHUNK_SIZE = prompt_config["parameters"]["chunk_size"]
CHUNK_OVERLAP = prompt_config["parameters"]["chunk_overlap"]

print(f"Using chunk_size={CHUNK_SIZE}, chunk_overlap={CHUNK_OVERLAP} "
      f"(from {PROMPT_CONFIG_PATH})\n")

# -------------------------------------------------
# Embedding Model
# -------------------------------------------------

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={
        "normalize_embeddings": True
    },
)

# -------------------------------------------------
# Tokenizer & Text Splitter
# -------------------------------------------------

tokenizer = AutoTokenizer.from_pretrained(
    "BAAI/bge-base-en-v1.5"
)

text_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
)

all_chunks = []

# -------------------------------------------------
# Load PDFs
# -------------------------------------------------

print("=" * 70)
print("Loading PDF Documents")
print("=" * 70)

if os.path.exists(PDF_FOLDER):

    for filename in os.listdir(PDF_FOLDER):

        if not filename.endswith(".pdf"):
            continue

        pdf_path = os.path.join(PDF_FOLDER, filename)

        loader = PyPDFLoader(pdf_path)

        docs = loader.load()

        chunks = text_splitter.split_documents(docs)

        # Assign a paragraph number = position of this chunk among all
        # chunks belonging to the same page, so citations can say
        # "Page 4, Paragraph 2" instead of just "Page 4".
        page_paragraph_counters = {}

        for chunk in chunks:
            page_num = chunk.metadata.get("page", 0)
            page_paragraph_counters[page_num] = page_paragraph_counters.get(page_num, 0) + 1
            chunk.metadata["paragraph"] = page_paragraph_counters[page_num]

        all_chunks.extend(chunks)

        print(f"{filename}")
        print(f"Pages extracted : {len(docs)}")
        print(f"Chunks created  : {len(chunks)}\n")

else:
    print("PDF folder not found.")

# -------------------------------------------------
# Load Web Pages
# -------------------------------------------------

print("=" * 70)
print("Loading Web Pages")
print("=" * 70)

if os.path.exists(WEBSITE_FILE):

    with open(WEBSITE_FILE, "r", encoding="utf-8") as f:

        urls = [
            line.strip()
            for line in f
            if line.strip()
        ]

    for url in urls:

        try:

            print(f"Loading: {url}")

            loader = WebBaseLoader(url)

            docs = loader.load()

            chunks = text_splitter.split_documents(docs)

            # Web pages have no "page" field, so paragraph numbers are
            # just sequential across the whole page's chunks.
            for i, chunk in enumerate(chunks, start=1):
                chunk.metadata["paragraph"] = i

            all_chunks.extend(chunks)

            print(f"Chunks created : {len(chunks)}\n")

        except Exception as e:

            print(f"Failed to load {url}")
            print(e)
            print()

else:
    print("websites.txt not found.\n")

# -------------------------------------------------
# Create Chroma Database
# -------------------------------------------------

if os.path.exists(CHROMA_PATH):
    print("Existing database found. Updating database...")
else:
    print("Creating new database...")

vectordb = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH,
)

# -------------------------------------------------
# Summary
# -------------------------------------------------

print("=" * 70)
print("Ingestion Complete")
print("=" * 70)

print(f"Total chunks stored : {len(all_chunks)}")
print(f"Database location   : {CHROMA_PATH}")
print(f"Chunk size / overlap : {CHUNK_SIZE} / {CHUNK_OVERLAP}")