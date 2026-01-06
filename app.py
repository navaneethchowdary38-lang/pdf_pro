import streamlit as st
from pypdf import PdfReader
import re
import numpy as np

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="Universal PDF QA", layout="wide")
st.title("📄 Universal PDF Question Answering")
st.write("Ask ANY question. Answers come ONLY from the PDF.")

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
def extract_text(pdf):
    reader = PdfReader(pdf)
    text = ""
    for page in reader.pages:
        if page.extract_text():
            text += page.extract_text() + " "
    return re.sub(r"\s+", " ", text)

def split_into_sentences(text):
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if len(s.strip()) > 30]

def embed_sentences(sentences):
    return model.encode(sentences)

def search_answer(question, sentences, embeddings, top_k=5):
    q_emb = model.encode([question])
    scores = cosine_similarity(q_emb, embeddings)[0]
    top_idx = np.argsort(scores)[::-1][:top_k]
    return [sentences[i] for i in top_idx if scores[i] > 0.35]

# -----------------------------------
# MAIN LOGIC
# -----------------------------------
if pdf_file:
    with st.spinner("Processing PDF..."):
        text = extract_text(pdf_file)
        sentences = split_into_sentences(text)
        embeddings = embed_sentences(sentences)

    st.success(f"PDF loaded successfully ({len(sentences)} sentences indexed)")

    question = st.text_input("Ask a question")

    if question:
        answers = search_answer(question, sentences, embeddings)

        st.subheader("Answer (Exact text from PDF)")

        if answers:
            for ans in answers:
                st.write(f"- {ans}")
        else:
            st.warning("No exact answer found in the document.")

else:
    st.info("Upload a PDF to start.")
