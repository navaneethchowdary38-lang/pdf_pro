import streamlit as st
from pypdf import PdfReader
import re

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="PDF QA (Question Enabled)", layout="wide")
st.title("📄 PDF Question Answering")
st.write("Ask questions and get EXACT answers from the PDF.")

# -----------------------------------
# PDF UPLOAD
# -----------------------------------
pdf_file = st.file_uploader(
    "Upload PDF file",
    type=["pdf"]
)

# -----------------------------------
# TEXT EXTRACTION
# -----------------------------------
def extract_full_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + "\n"
    return re.sub(r"\s+", " ", text)

# -----------------------------------
# SECTION EXTRACTION
# -----------------------------------
def extract_section(text, start, end_list):
    pattern = start + r"(.*?)(" + "|".join(end_list) + r")"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return match.group(1).strip()
    return None

# -----------------------------------
# MAIN LOGIC
# -----------------------------------
if pdf_file:
    text = extract_full_text(pdf_file)

    question = st.text_input("Ask a question")

    if question:
        q = question.lower()

        if "course outcome" in q:
            answer = extract_section(
                text,
                "COURSE OUTCOMES:",
                ["EXPERIMENTS:", "COURSE OBJECTIVES:"]
            )

        elif "course objective" in q:
            answer = extract_section(
                text,
                "COURSE OBJECTIVES:",
                ["COURSE OUTCOMES:", "EXPERIMENTS:"]
            )

        elif "experiment" in q:
            answer = extract_section(
                text,
                "EXPERIMENTS:",
                ["COURSE OUTCOMES:", "COURSE OBJECTIVES:"]
            )

        else:
            answer = None

        st.subheader("Answer")

        if answer:
            points = re.findall(r"\d+\.\s.*?(?=\d+\.|$)", answer)
            if points:
                for p in points:
                    st.write(f"- {p.strip()}")
            else:
                st.write(answer)
        else:
            st.warning("Question not supported. Try asking about course outcomes, objectives, or experiments.")

else:
    st.info("Please upload a PDF to start.")
