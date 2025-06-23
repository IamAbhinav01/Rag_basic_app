import os
import streamlit as st
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_cohere import CohereEmbeddings
from langchain.vectorstores import FAISS
from langchain_cohere import ChatCohere
from langchain_core.documents import Document
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain

# Set Cohere key from Hugging Face Secrets
COHERE_API_KEY = st.secrets["COHERE_API_KEY"]
# Initialize LLM
llm = ChatCohere(cohere_api_key=COHERE_API_KEY)

# Streamlit UI
st.set_page_config(page_title="PDF Q&A with LangChain + Cohere")
st.title("📄 Ask Questions from Your PDF (Cohere + LangChain)")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")
question = st.text_input("Enter your question")

if uploaded_file and question:
    # Save uploaded file temporarily
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    # Load and split document
    loader = PyPDFLoader("temp.pdf")
    docs = loader.load()

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    # Embeddings + FAISS
    embeddings = CohereEmbeddings(model="embed-english-v3.0", cohere_api_key=COHERE_API_KEY)
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever(search_kwargs={"k": 3})

    # Prompt template
    prompt = ChatPromptTemplate.from_template("""
    You are an AI assistant. Use the provided context to answer the question accurately and concisely.
    If the answer is not in the context, say "The information is not available in the provided context."

    <context>
    {context}
    </context>

    Question: {input}

    Answer:
    """)

    # RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # Run RAG
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": question})
        st.subheader("📌 Answer")
        st.write(response['answer'])
