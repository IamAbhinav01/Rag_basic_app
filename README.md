# 📄 PDF Q&A App using LangChain + Cohere

This project demonstrates a simple Retrieval-Augmented Generation (RAG) application that allows users to upload a PDF and ask questions based on its content. It uses LangChain, FAISS, and Cohere's embedding and language models.

---

## 🚀 Features

- Upload any PDF document
- Ask natural language questions
- Answers are grounded in the document content
- Uses user-provided API key for secure, zero-cost use
- Clean Streamlit interface for interaction

---

## 🛠️ Tech Stack

- [LangChain](https://www.langchain.com/) for prompt management and chains
- [Cohere](https://cohere.com/) for embeddings & LLM generation
- [FAISS](https://github.com/facebookresearch/faiss) for semantic search
- [Streamlit](https://streamlit.io/) for web interface

---

## 🔐 Bring Your Own Cohere API Key

To use the app:
1. Visit [https://dashboard.cohere.com](https://dashboard.cohere.com) and create a free API key
2. Enter the key in the app when prompted
3. Upload your PDF and ask questions!

> This ensures your key is used only in your own session and avoids hardcoding secrets.

---

## 📦 Setup Locally

```bash
git clone https://github.com/IamAbhinav01/Rag_basic_app.git
cd Rag_basic_app
pip install -r requirements.txt
streamlit run app.py

🌐 Live Demo
👉 : https://lnkd.in/gKqpjqZx

