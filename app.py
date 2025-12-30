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
st.write("Answers are extracted EXACTLY from the PDF (no hallucination).")

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
        chunk_size=700,      # slightly larger for lists
        chunk_overlap=150
    )
    return splitter.split_text(text)


def build_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


# -----------------------------------
# LOAD EXTRACTIVE QA MODEL
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

    question = st.text_input("Ask a question (use wording from the PDF)")

    if question:
        # retrieve more chunks to avoid missing list items
        docs = vector_store.similarity_search(question, k=8)

        answers = []

        for doc in docs:
            result = qa_model(
                question=question,
                context=doc.page_content
            )

            # LOWER threshold – extractive models are conservative
            if result["score"] > 0.05:
                answers.append(result["answer"].strip())

        # remove duplicates while preserving order
        answers = list(dict.fromkeys(answers))

        st.subheader("Answer (from PDF)")

        if answers:
            for ans in answers:
                st.write(f"- {ans}")
        else:
            st.write("Answer not found in the document.")

else:
    st.info("Please upload at least one PDF.")
