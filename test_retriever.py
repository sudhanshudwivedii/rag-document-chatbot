from pathlib import Path

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Project directory
BASE_DIR = Path(__file__).resolve().parent
DB_DIR = BASE_DIR / "chroma_db"

# 1. Initialize the same embedding model
embedding_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

# 2. Connect to existing ChromaDB
vectorstore = Chroma(
    collection_name="rag_documents",
    persist_directory=str(DB_DIR),
    embedding_function=embedding_model
)

# 3. Create Retriever
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

# 4. Test retrieval
query = "What is LangChain?"

documents = retriever.invoke(query)

# 5. Display retrieved documents
for doc in documents:
    print(doc.page_content)
    print("Source:", doc.metadata)
    print("-" * 50)