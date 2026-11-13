import os
from hashlib import md5
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

# ---------------------------------------------------------------- config
BASE_DIR = Path(__file__).resolve().parent
DOCUMENTS_DIR = BASE_DIR / "documents"
DB_DIR = BASE_DIR / "chroma_db"
COLLECTION_NAME = "rag_documents"  # same name used in create_database.py
DOCUMENTS_DIR.mkdir(exist_ok=True)

st.set_page_config(page_title="RAG Project", page_icon="📚", layout="centered")

if not os.getenv("GROQ_API_KEY"):
    st.error("GROQ_API_KEY is missing. Add it to your .env file and restart the app.")
    st.stop()

PROMPT = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "You are a helpful AI assistant.\n\n"
            "Use ONLY the provided context to answer the question.\n\n"
            "If the answer is not present in the context, "
            'say: "I could not find the answer in the document."',
        ),
        ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
    ]
)

SPLITTER = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)


# ---------------------------------------------------------------- cached resources
@st.cache_resource(show_spinner="Loading embedding model...")
def get_vectorstore() -> Chroma:
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return Chroma(
        collection_name=COLLECTION_NAME,
        embedding_function=embeddings,
        persist_directory=str(DB_DIR),
    )


@st.cache_resource
def get_llm() -> ChatGroq:
    return ChatGroq(model="openai/gpt-oss-20b", temperature=0.2)


# ---------------------------------------------------------------- helpers
def list_indexed_files(vs: Chroma) -> list[str]:
    metadatas = vs.get(include=["metadatas"])["metadatas"] or []
    return sorted({m["source_name"] for m in metadatas if m and "source_name" in m})


def index_pdf(vs: Chroma, uploaded_file) -> int:
    """Save the PDF, split it into chunks and add them to ChromaDB."""
    file_name = Path(uploaded_file.name).name
    data = uploaded_file.getvalue()

    pdf_path = DOCUMENTS_DIR / file_name
    pdf_path.write_bytes(data)

    pages = PyPDFLoader(str(pdf_path)).load()
    for page in pages:
        page.metadata["source_name"] = file_name

    chunks = SPLITTER.split_documents(pages)
    if not chunks:  # e.g. scanned PDF with no text layer
        return 0

    file_id = md5(data).hexdigest()[:10]
    vs.add_documents(chunks, ids=[f"{file_id}-{i}" for i in range(len(chunks))])
    return len(chunks)


def clear_database(vs: Chroma) -> None:
    ids = vs.get()["ids"]
    if ids:
        vs.delete(ids=ids)


def ask(vs: Chroma, question: str):
    retriever = vs.as_retriever(
        search_type="mmr",
        search_kwargs={"k": 4, "fetch_k": 10, "lambda_mult": 0.5},
    )
    docs = retriever.invoke(question)
    context = "\n\n".join(doc.page_content for doc in docs)
    response = get_llm().invoke(PROMPT.invoke({"context": context, "question": question}))
    sources = [
        {
            "file": d.metadata.get("source_name", "unknown"),
            "page": int(d.metadata.get("page", 0)) + 1,
            "text": d.page_content,
        }
        for d in docs
    ]
    return response.content, sources


def show_sources(sources: list[dict]) -> None:
    with st.expander(f"Sources ({len(sources)})"):
        for s in sources:
            st.markdown(f"**{s['file']}**, page {s['page']}")
            st.caption(s["text"][:400] + ("..." if len(s["text"]) > 400 else ""))


# ---------------------------------------------------------------- page
vectorstore = get_vectorstore()
indexed_files = list_indexed_files(vectorstore)
has_docs = bool(indexed_files)

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("📚 RAG Project")
st.subheader("Upload your file and ask me anything about it")

# --- Step 1: upload
with st.expander("📄 Upload your PDF", expanded=not has_docs):
    uploads = st.file_uploader(
        "Choose one or more PDF files",
        type="pdf",
        accept_multiple_files=True,
    )

    if st.button("Process documents", type="primary", disabled=not uploads):
        with st.spinner("Reading and indexing your documents..."):
            for f in uploads:
                if f.name in indexed_files:
                    st.info(f"{f.name} is already indexed.")
                    continue
                try:
                    n = index_pdf(vectorstore, f)
                except Exception as e:
                    st.error(f"Could not process {f.name}: {e}")
                    continue
                if n:
                    st.success(f"{f.name}: added {n} chunks.")
                else:
                    st.warning(
                        f"{f.name}: no readable text found. "
                        "Scanned PDFs need OCR before they can be indexed."
                    )
        st.rerun()

# --- Indexed documents
if has_docs:
    st.markdown("**Ready to answer questions about:**")
    for name in indexed_files:
        st.markdown(f"- {name}")

    col1, col2 = st.columns(2)
    if col1.button("Clear chat"):
        st.session_state.messages = []
        st.rerun()
    if col2.button("Remove all documents"):
        clear_database(vectorstore)
        st.session_state.messages = []
        st.rerun()
else:
    st.info("No documents yet. Upload a PDF above and click **Process documents** to start.")

st.divider()

# --- Step 2: chat
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            show_sources(msg["sources"])

question = st.chat_input(
    "Ask a question about your document...",
    disabled=not has_docs,
)

if question:
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching your documents..."):
            try:
                answer, sources = ask(vectorstore, question)
            except Exception as e:
                answer, sources = f"Something went wrong: {e}", []
        st.markdown(answer)
        if sources:
            show_sources(sources)

    st.session_state.messages.append(
        {"role": "assistant", "content": answer, "sources": sources}
    )