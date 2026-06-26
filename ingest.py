from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
# Step 1: Load the PDF
loader = PyPDFLoader("document.pdf")
docs = loader.load()
print(f"Total pages loaded: {len(docs)}")

# Step 2: Split into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=700,
    chunk_overlap=100
)
chunks = splitter.split_documents(docs)
print(f"Total chunks created: {len(chunks)}")

# Step 3: Preview first chunk
print("\n--- First Chunk ---")
print(chunks[0].page_content)
print("\n--- Metadata ---")
print(chunks[0].metadata)