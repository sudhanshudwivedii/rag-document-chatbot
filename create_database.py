
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Project directories
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DB_DIR = BASE_DIR / "chroma_db"

# 1. Initialize embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Load PDF documents
pdf_files = list(DOCUMENTS_DIR.glob("*.pdf"))

if not pdf_files:
    raise FileNotFoundError(
        "No PDF files found inside the documents folder."
    )

documents = []

for pdf_file in pdf_files:
    loader = PyPDFLoader(str(pdf_file))
    documents.extend(loader.load())

# 3. Split documents into chunks
splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

chunks = splitter.split_documents(documents)

# 4. Create and store embeddings in ChromaDB
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embedding_model,
    persist_directory=str(DB_DIR),
    collection_name="rag_documents",
    ids=[f"chunk-{i}" for i in range(len(chunks))]
)

print(f"Loaded {len(pdf_files)} PDF files.")
print(f"Created {len(chunks)} document chunks.")
print("Vector database created successfully!")
