import streamlit as st
from pypdf import PdfReader
import re

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="PDF QA (Section Accurate)", layout="wide")
st.title("📄 PDF Question Answering (Section Accurate)")
st.write("Answers are extracted EXACTLY from the correct section of the PDF.")

# -----------------------------------
# PDF UPLOAD
# -----------------------------------
pdf_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=False
)

# -----------------------------------
# TEXT EXTRACTION
# -----------------------------------
def extract_full_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return re.sub(r"\s+", " ", text)

# -----------------------------------
# SECTION EXTRACTION (KEY FIX)
# -----------------------------------
def extract_course_outcomes(text):
    pattern = r"COURSE OUTCOMES:(.*?)(EXPERIMENTS:|COURSE OBJECTIVES:)"
    match = re.search(pattern, text, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).strip()

# -----------------------------------
# MAIN LOGIC
# -----------------------------------
if pdf_files:
    with st.spinner("Reading PDF..."):
        text = extract_full_text(pdf_files)
        course_outcomes = extract_course_outcomes(text)

    st.subheader("Question Example")
    st.code("What are the course outcomes?")

    st.subheader("Answer (Exact from PDF)")

    if course_outcomes:
        # Clean numbered formatting
        outcomes = re.findall(r"\d+\.\s.*?(?=\d+\.|$)", course_outcomes)
        for o in outcomes:
            st.write(f"- {o.strip()}")
    else:
        st.error("COURSE OUTCOMES section not found in this PDF.")

else:
    st.info("Please upload a PDF.")
