import streamlit as st
from pypdf import PdfReader
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="Universal PDF QA (Improved)", layout="wide")
st.title("📄 Universal PDF Question Answering")
st.write("Works for ALL PDFs. Answers are taken ONLY from the PDF.")

# -----------------------------------
# LOAD MODEL
# -----------------------------------
@st.cache_resource
def load_model():
    return SentenceTransformer("all-MiniLM-L6-v2")

model = load_model()

# -----------------------------------
# PDF UPLOAD
# -----------------------------------
pdf_file = st.file_uploader("Upload any PDF", type=["pdf"])

# -----------------------------------
# FUNCTIONS
# -----------------------------------
def extract_lines(pdf):
    reader = PdfReader(pdf)
    lines = []
    for page in reader.pages:
        text = page.extract_text()
        if text:
            for line in text.split("\n"):
                line = re.sub(r"\s+", " ", line).strip()
                if len(line) > 5:
                    lines.append(line)
    return lines


def is_section_header(line):
    return (
        line.isupper()
        or line.endswith(":")
        or re.match(r"^\d+(\.\d+)*\s", line)
        or re.search(r"(OUTCOMES|OBJECTIVES|UNIT|EXPERIMENTS|COURSE|SYLLABUS)", line, re.I)
    )


def build_sections(lines):
    sections = {}
    current_section = "GENERAL"

    for line in lines:
        if is_section_header(line):
            current_section = line
            sections.setdefault(current_section, [])
        else:
            sections.setdefault(current_section, []).append(line)

    return sections


def flatten_sections(sections):
    texts = []
    meta = []

    for section, lines in sections.items():
        paragraph = " ".join(lines)
        sentences = re.split(r'(?<=[.!?])\s+', paragraph)

        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 30:
                texts.append(sent)
                meta.append(section)

    return texts, meta


def embed_texts(texts):
    return model.encode(texts)


def search_answer(question, texts, meta, embeddings, top_k=6):
    q_emb = model.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]

    ranked = np.argsort(scores)[::-1]

    answers = []
    seen = set()

    for i in ranked:
        sentence = texts[i]
        section = meta[i]

        if len(sentence) < 30:
            continue

        key = sentence.lower()
        if key in seen:
            continue

        answers.append((section, sentence))
        seen.add(key)

        if len(answers) == top_k:
            break

    return answers

# -----------------------------------
# MAIN LOGIC
# -----------------------------------
if pdf_file:
    with st.spinner("Analyzing PDF and building index..."):
        lines = extract_lines(pdf_file)
        sections = build_sections(lines)
        texts, meta = flatten_sections(sections)
        embeddings = embed_texts(texts)

    st.success(f"Indexed {len(texts)} statements from {len(sections)} sections")

    question = st.text_input("Ask a question")

    if question:
        answers = search_answer(question, texts, meta, embeddings)

        st.subheader("Answer (Exact text from PDF)")

        if answers:
            for section, text in answers:
                st.markdown(f"**{section}**")
                st.write(f"- {text}")
        else:
            st.warning("No relevant answer found in the document.")

else:
    st.info("Upload a PDF to begin.")
