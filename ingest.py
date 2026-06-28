import os

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings

# ----------------------------
# Configuration
# ----------------------------

PDF_FOLDER = "./documents"
CHROMA_PATH = "./chroma_db"

embedding_model = HuggingFaceEmbeddings(
    model_name="BAAI/bge-base-en-v1.5",
    encode_kwargs={"normalize_embeddings": True},
)

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)

all_chunks = []



for filename in os.listdir(PDF_FOLDER):

    if not filename.endswith(".pdf"):
        continue

    pdf_path = os.path.join(PDF_FOLDER, filename)

    

    loader = PyPDFLoader(pdf_path)

    docs = loader.load()

    print(f"\n{filename}")
    print(f"Pages extracted: {len(docs)}")


    chunks = text_splitter.split_documents(docs)


    all_chunks.extend(chunks)
    print(filename, len(chunks))

    print(f"Chunks: {len(chunks)}")


if os.path.exists(CHROMA_PATH):
    print("Existing database found.")
else:
    print("Creating new database.")

vectordb = Chroma.from_documents(
    documents=all_chunks,
    embedding=embedding_model,
    persist_directory=CHROMA_PATH,
)

print("\nDone!")

print(f"Stored {len(all_chunks)} chunks.")

print(f"Database location: {CHROMA_PATH}")