import streamlit as st
from PyPDF2 import PdfReader
from dotenv import load_dotenv
import os

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_community.vectorstores import FAISS
from langchain.chains import RetrievalQA

# -----------------------------------
# ENV SETUP
# -----------------------------------
load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

# -----------------------------------
# STREAMLIT UI
# -----------------------------------
st.set_page_config(page_title="PDF Analyzer Chatbot", layout="wide")
st.title("📄 PDF Analyzer Chatbot")
st.write("Upload PDFs and ask questions based on their content.")

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
            text += page.extract_text() or ""
    return text


def split_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    return splitter.split_text(text)


def create_vector_store(chunks):
    embeddings = GoogleGenerativeAIEmbeddings(
        model="models/embedding-001"
    )
    return FAISS.from_texts(chunks, embeddings)


def create_qa_chain(vector_store):
    llm = ChatGoogleGenerativeAI(
        model="gemini-pro",
        temperature=0.3
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
        chunks = split_text(raw_text)
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

