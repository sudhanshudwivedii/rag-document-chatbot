
# RAG-Based Document Chatbot

An AI-powered document question-answering application built using Retrieval-Augmented Generation (RAG).

Users can upload PDF documents and ask questions. The application retrieves relevant information from the documents and generates context-aware answers using an LLM.

## Features

- PDF document upload
- Document loading and text extraction
- Text chunking
- Hugging Face embeddings
- ChromaDB vector storage
- Semantic similarity search
- Context-based answer generation
- Source document references
- Interactive Streamlit interface

## Tech Stack

- Python
- LangChain
- Groq API
- Hugging Face
- ChromaDB
- Streamlit
- UV

## Architecture

PDF Upload → Document Loader → Text Splitter → Embeddings → ChromaDB → Retriever → Groq LLM → Answer

## Installation

1. Clone the repository.
2. Create a virtual environment using UV.
3. Install dependencies:

   uv pip install -r requirements.txt

4. Create a `.env` file and add your Groq API key.
5. Run the application:

   streamlit run app.py

## Future Improvements

- Support for multiple document formats
- Advanced retrieval strategies
- Improved source citations
- Conversation memory
- RAG evaluation
