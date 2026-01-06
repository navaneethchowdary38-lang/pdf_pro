import streamlit as st
from pypdf import PdfReader
import re

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="PDF QA (Exact & Accurate)", layout="wide")
st.title("📄 PDF Question Answering (Exact)")
st.write("Answers are returned EXACTLY as written in the PDF.")

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


def split_into_sentences(text):
    # Clean spacing
    text = re.sub(r"\s+", " ", text).strip()

    # Sentence split
    sentences = re.split(r'(?<=[.!?])\s+', text)

    # Filter very short lines
    return [s.strip() for s in sentences if len(s.strip()) > 40]


def build_vector_store(sentences):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(sentences, embeddings)


# -----------------------------------
# MAIN LOGIC
# -----------------------------------
if pdf_files:
    with st.spinner("Processing PDFs..."):
        raw_text = extract_text_from_pdfs(pdf_files)

        if not raw_text.strip():
            st.error("No text found in PDFs.")
            st.stop()

        sentences = split_into_sentences(raw_text)
        vector_store = build_vector_store(sentences)

    st.success("PDF indexed successfully!")

    question = st.text_input(
        "Ask a question (use wording similar to the PDF)"
    )

    if question:
        # Retrieve many sentences to avoid missing list items
        results = vector_store.similarity_search(question, k=20)

        answers = []
        seen = set()

        for r in results:
            sentence = r.page_content.strip()
            if sentence not in seen:
                seen.add(sentence)
                answers.append(sentence)

        st.subheader("Answer (Exact text from PDF)")

        if answers:
            for ans in answers:
                st.write(f"- {ans}")
        else:
            st.write("No matching text found.")

else:
    st.info("Please upload at least one PDF.")
