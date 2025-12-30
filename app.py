import streamlit as st
from PyPDF2 import PdfReader
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI
)
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="PDF Analyzer Chatbot", layout="wide")
st.title("📄 PDF Analyzer Chatbot")
st.write("Upload PDFs and ask questions based on their content.")

# -----------------------------------
# API KEY (Streamlit-safe)
# -----------------------------------
# ❌ Do NOT use load_dotenv() on Streamlit Cloud
# ✅ Use st.secrets instead

GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY")

if not GOOGLE_API_KEY:
    st.error("GOOGLE_API_KEY not found. Please set it in Streamlit Secrets.")
    st.stop()

# -----------------------------------
# PDF UPLOAD
# -----------------------------------
pdf_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------------
# FUNCTIONS
# -----------------------------------
def extract_text_from_pdfs(pdfs):
    text = ""
    for pdf in pdfs:
        reader = PdfReader(pdf)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1500,
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    # remove empty chunks
    return [chunk for chunk in chunks if chunk.strip()]


def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="text-embedding-004",
        google_api_key=GOOGLE_API_KEY
    )
    return FAISS.from_texts(chunks, embeddings)


def create_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3,
        google_api_key=GOOGLE_API_KEY
    )
    return RetrievalQA.from_chain_type(
        llm=llm,
        retriever=vector_store.as_retriever()
    )

# -----------------------------------
# MAIN LOGIC
# -----------------------------------
if pdf_files:
    with st.spinner("Processing PDFs..."):
        raw_text = extract_text_from_pdfs(pdf_files)

        if not raw_text.strip():
            st.error("No text could be extracted from the PDFs.")
            st.stop()

        chunks = split_text(raw_text)

        if len(chunks) == 0:
            st.error("Text splitting failed. No chunks created.")
            st.stop()

        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)

    st.success("PDFs processed successfully!")

    question = st.text_input("Ask a question from the PDFs")

    if question:
        with st.spinner("Generating answer..."):
            response = qa_chain.run(question)

        st.subheader("Answer")
        st.write(response)

else:
    st.info("Please upload at least one PDF to start.")
