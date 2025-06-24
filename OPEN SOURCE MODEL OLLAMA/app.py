import os
import streamlit as st
from dotenv import load_dotenv
load_dotenv()

os.environ["LANGSMITH_API_KEY"] = os.getenv("LANGSMITH_API_KEY")
os.environ["LANGSMITH_TRACING"] = "true"
os.environ["LANGSMITH_PROJECT"] = os.getenv("LANGSMITH_PROJECT")

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_ollama import OllamaEmbeddings
from langchain_ollama import ChatOllama
from langchain.vectorstores import FAISS
from langchain.schema import Document
from langchain_core import retrievers
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
st.set_page_config(page_title="PDF Q&A with LangChain + Cohere")
st.title("📄 Ask Questions from Your PDF (llama3 + LangChain)")
uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

input_text=st.text_input("What question you have in mind?")

if uploaded_file and input_text:
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
    embeddings = OllamaEmbeddings(model="llama3")
    db = FAISS.from_documents(chunks, embeddings)
    retriever = db.as_retriever()

    # Prompt template
    prompt = ChatPromptTemplate.from_template(
    
        """ 
            You are an AI assistant. Use the provided context to answer the question accurately and concisely.
        If the answer is not in the context, say "The information is not available in the provided context."

        <context>
        {context}
        </context>

        Question: {input}
        Answer:
        """

    
)
    llm = ChatOllama(model="llama3")
    # RAG chain
    document_chain = create_stuff_documents_chain(llm, prompt)
    rag_chain = create_retrieval_chain(retriever, document_chain)

    # Run RAG
    with st.spinner("Thinking..."):
        response = rag_chain.invoke({"input": input_text})
        st.subheader("📌 Answer")
        st.write(response['answer'])
