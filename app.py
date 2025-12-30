import streamlit as st
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.llms import HuggingFaceHub
from langchain.chains import RetrievalQA

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="PDF Analyzer Chatbot", layout="wide")
st.title("📄 PDF Analyzer Chatbot (Free & Stable)")
st.write("Upload PDFs and ask questions.")

# -----------------------------------
# CHECK HF TOKEN
# -----------------------------------
if "HUGGINGFACEHUB_API_TOKEN" not in st.secrets:
    st.error("HuggingFace API token not found in Streamlit Secrets.")
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
        chunk_size=1000,
        chunk_overlap=150
    )
    chunks = splitter.split_text(text)
    return [c for c in chunks if len(c.strip()) > 50]


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


def create_qa_chain(vector_store):
    llm = HuggingFaceHub(
        repo_id="google/flan-t5-base",
        model_kwargs={
            "temperature": 0.3,
            "max_length": 512
        }
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
            st.error("No text could be extracted from PDFs.")
            st.stop()

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
