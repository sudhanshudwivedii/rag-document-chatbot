# 📚 RAG-Based Document Chatbot

### Chat with your PDF documents using Generative AI

An AI-powered document question-answering application built using **Retrieval-Augmented Generation (RAG)**. Upload PDF documents, ask questions, and receive context-aware answers based on the information in your documents.

<p align="center">

**🚀 [LIVE DEMO – TRY THE CHATBOT](https://sudhanshu-rag-chatbot.streamlit.app)**

</p>

---

## 🌐 Live Application

🔗 **Try it here:** https://sudhanshu-rag-chatbot.streamlit.app

💻 **GitHub Repository:** https://github.com/sudhanshudwivedi/rag-document-chatbot

Upload a PDF, ask a question about its content, and explore the retrieved source information.

## ✨ Key Features

- 📄 Upload one or multiple PDF documents.
- 📝 Extract text from uploaded documents.
- ✂️ Split documents into smaller text chunks.
- 🧠 Generate embeddings using Hugging Face.
- 🗄️ Store document embeddings in ChromaDB.
- 🔍 Retrieve relevant information using semantic similarity search.
- 🤖 Generate context-aware answers using Groq LLM.
- 📚 Display relevant source references.
- 💬 Interactive chatbot interface built with Streamlit.

## 🏗️ RAG Architecture

The application follows this workflow:

**PDF Upload → Document Loading → Text Splitting → Embedding Generation → ChromaDB Vector Storage → Retriever → Groq LLM → Final Answer**

### How It Works

1. **Document Upload:** Users upload PDF files through the Streamlit interface.
2. **Document Processing:** Text is extracted and divided into smaller chunks.
3. **Embedding Generation:** Hugging Face converts text chunks into numerical vectors.
4. **Vector Storage:** ChromaDB stores the embeddings and document information.
5. **Information Retrieval:** The retriever finds relevant document chunks based on the user's question.
6. **Answer Generation:** Groq LLM uses the retrieved context to generate a response.
7. **Source References:** The application displays relevant document sources.

## 🛠️ Tech Stack

| Technology | Purpose |
|---|---|
| Python | Core programming language |
| LangChain | LLM and RAG integration |
| Groq API | LLM-powered answer generation |
| Hugging Face | Text embedding generation |
| ChromaDB | Vector database |
| Streamlit | Web application interface |
| PyPDF | PDF document loading |
| UV | Python package and environment management |

## 📂 Project Structure

```text
RAG-Document-Chatbot/
│
├── app.py
├── main.py
├── create_database.py
├── test_retriever.py
│
├── document loader/
├── Vector Store/
│
├── requirements.txt
├── README.md
├── .gitignore
└── .env.example
```

## ⚙️ Run Locally

### 1. Clone the repository

Use the repository's GitHub Code → HTTPS option to copy its clone URL.

```bash
git clone YOUR_REPOSITORY_URL
```

### 2. Navigate to the project

```bash
cd rag-document-chatbot
```

### 3. Create a virtual environment

```bash
uv venv --python 3.12
```

### 4. Activate the environment

```bash
source .venv/bin/activate
```

### 5. Install dependencies

```bash
uv pip install -r requirements.txt
```

### 6. Configure your API key

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
```

Get your API key from [Groq Console](https://console.groq.com/keys).

### 7. Run the application

```bash
streamlit run app.py
```

Open the local URL displayed in your terminal, usually:

```text
http://localhost:8501
```

## 🔐 Security

- API keys are stored in environment variables.
- The `.env` file is excluded from Git.
- Never commit private API keys or sensitive documents.

## 🚀 Future Improvements

- Support for additional document formats.
- Advanced retrieval techniques and reranking.
- Improved document source citations.
- Conversation memory.
- RAG evaluation and performance monitoring.
- Multi-user document management.

## 👨‍💻 Author

**Sudhanshu Dwivedi**

Computer Science Engineering | Artificial Intelligence & Machine Learning

Interested in Generative AI, Machine Learning, RAG Systems and AI-powered applications.

🔗 **GitHub:** https://github.com/sudhanshudwivedi

🔗 **LinkedIn:** Add your LinkedIn profile URL here.

---

⭐ If you find this project useful, consider giving the repository a star!