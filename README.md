# 🤖 RAG Chatbot (Semantic Search + Groq + Streamlit)

This project is a **Retrieval-Augmented Generation (RAG) Chatbot** that answers user queries by retrieving relevant information from a custom document and generating responses using an LLM.

It combines **semantic search + embeddings + LLM reasoning** to produce accurate, context-based answers.

---

## 🚀 Live Application

🔗 https://rag-chatbot-endee-2jtnf7dyhzpmpdwkd7ztuh.streamlit.app/

---

## 🧠 How the System Works

The chatbot follows a simple but powerful pipeline:

1. 📄 Documents are stored in a text file (`docs.txt`)
2. 🧠 Each document is converted into embeddings using SentenceTransformers
3. 🔍 When a user asks a question:
   - The query is also converted into an embedding
   - Cosine similarity is used to find the most relevant documents
4. 📚 Top matching documents are selected as context
5. 🤖 Context + question is sent to Groq LLM
6. 💬 LLM generates a final answer based only on retrieved context

---

## ✨ Features

- Semantic search (not keyword-based)
- Context-aware responses (reduces hallucination)
- Fast LLM inference using Groq API
- Simple and interactive Streamlit UI
- Secure API key handling (Streamlit Secrets)
- Lightweight and deployable project

---

## 🛠️ Tech Stack

- **Python**
- **Streamlit** (Frontend + Deployment)
- **SentenceTransformers** (`all-MiniLM-L6-v2`)
- **Groq API (LLaMA 3.1)** for response generation
- **NumPy** for similarity calculation

## 💡 Example Queries

- What is semantic search?
- What is RAG?
- What is machine learning?
- Explain vector databases

---

## 🎯 Key Learnings

- Understanding of RAG architecture
- Working with embeddings and vector similarity
- Integrating LLM APIs (Groq)
- Building and deploying ML apps with Streamlit
- Handling real-world deployment issues (paths, secrets, caching)

---

## 🚀 Future Improvements

- Upload custom PDFs or documents
- Add chat history (memory)
- Use vector databases (FAISS / Pinecone)
- Improve UI (chat-style interface)
- Add source citations

---

## 🧾 Resume Description

Built and deployed a Retrieval-Augmented Generation (RAG) chatbot using SentenceTransformers, Groq LLM, and Streamlit, enabling semantic search and context-aware responses over custom documents.

---

## 👨‍💻 Author

**Sumit Kumar**  
GitHub: https://github.com/Sumitkr28

---

## ⭐ Support

If you found this useful, consider giving this repo a ⭐

