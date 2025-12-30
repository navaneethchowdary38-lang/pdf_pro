import streamlit as st
from pypdf import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import pipeline

# -----------------------------------
# STREAMLIT CONFIG
# -----------------------------------
st.set_page_config(page_title="PDF Analyzer Chatbot", layout="wide")
st.title("📄 PDF Analyzer Chatbot (Accurate & Free)")
st.write("Upload PDFs and ask questions based ONLY on their content.")

# -----------------------------------
# PDF UPLOAD
# -----------------------------------
pdf_files = st.file_uploader(
    "Upload PDF files",
    type=["pdf"],
    accept_multiple_files=True
)

# -----------------------------------
# STRICT QA PROMPT (VERY IMPORTANT)
# -----------------------------------
QA_PROMPT = PromptTemplate(
    input_variables=["context", "question"],
    template="""
You are a factual assistant.

Answer the question ONLY using the information provided in the context below.
Do NOT use outside knowledge.
If the answer is not clearly present in the context, say exactly:
"I cannot find the answer in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
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
        chunk_size=800,      # smaller chunks = better retrieval
        chunk_overlap=200
    )
    chunks = splitter.split_text(text)
    return [chunk for chunk in chunks if len(chunk.strip()) > 50]


def create_vector_store(chunks):
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-MiniLM-L3-v2"
    )
    return FAISS.from_texts(chunks, embeddings)


def create_qa_chain(vector_store):
    # Free, local, stable model
    hf_pipeline = pipeline(
        "text2text-generation",
        model="google/flan-t5-base",  # change to flan-t5-large if RAM allows
        max_length=512,
        temperature=0.0               # VERY IMPORTANT: reduces hallucinations
    )

    llm = HuggingFacePipeline(pipeline=hf_pipeline)

    retriever = vector_store.as_retriever(
        search_kwargs={"k": 6}         # fetch more relevant chunks
    )

    return RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        chain_type_kwargs={"prompt": QA_PROMPT}
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
        vector_store = create_vector_store(chunks)
        qa_chain = create_qa_chain(vector_store)

    st.success("PDFs processed successfully!")

    question = st.text_input("Ask a question from the PDFs")

    if question:
        with st.spinner("Generating accurate answer..."):
            answer = qa_chain.run(question)

        st.subheader("Answer")
        st.write(answer)

else:
    st.info("Please upload at least one PDF to start.")
