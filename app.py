import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from transformers import pipeline

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="PDF QA (Accurate)", layout="wide")
st.title("📄 PDF Question Answering (Accurate)")
st.write("Answers are extracted EXACTLY from the PDF.")

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
        chunk_size=500,      # small chunks for exact matching
        chunk_overlap=100
    )
    return splitter.split_text(text)


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


# -----------------------------------
# LOAD EXTRACTIVE QA MODEL (KEY PART)
# -----------------------------------
@st.cache_resource
def load_qa_model():
    return pipeline(
        "question-answering",
        model="deepset/roberta-base-squad2"
    )

qa_model = load_qa_model()

# -----------------------------------
# MAIN LOGIC
# -----------------------------------
if pdf_files:
    with st.spinner("Processing PDFs..."):
        raw_text = extract_text_from_pdfs(pdf_files)
        chunks = split_text(raw_text)
        vector_store = build_vector_store(chunks)

    st.success("PDFs processed successfully!")

    question = st.text_input("Ask a question")

    if question:
        # Retrieve most relevant chunks
        docs = vector_store.similarity_search(question, k=3)

        best_answer = ""
        best_score = 0

        for doc in docs:
            result = qa_model(
                question=question,
                context=doc.page_content
            )
            if result["score"] > best_score:
                best_score = result["score"]
                best_answer = result["answer"]

        st.subheader("Answer (from PDF)")
        if best_answer.strip():
            st.write(best_answer)
        else:
            st.write("Answer not found in the document.")

else:
    st.info("Please upload at least one PDF.")
