import streamlit as st
import PyPDF2
import docx
from io import BytesIO
import re

# Lazy model loading with caching
@st.cache_resource(max_entries=1, ttl=3600)
def load_lightweight_models():
    """Lightweight models for Streamlit Cloud"""
    from transformers import pipeline, AutoTokenizer, AutoModelForQuestionAnswering
    import torch
    
    # Use smaller, faster models
    qa_pipeline = pipeline(
        "question-answering",
        model="distilbert-base-cased-distilled-squad",  # Smaller model
        tokenizer="distilbert-base-cased-distilled-squad",
        device=-1  # CPU only
    )
    
    # Simple T5 summarizer (smaller)
    summarizer = pipeline(
        "summarization",
        model="sshleifer/distilbart-cnn-12-6",  # Much smaller
        tokenizer="sshleifer/distilbart-cnn-12-6"
    )
    
    return qa_pipeline, summarizer

def extract_text(uploaded_file):
    """Extract text from files"""
    if uploaded_file.type == "application/pdf":
        pdf_reader = PyPDF2.PdfReader(uploaded_file)
        return " ".join([page.extract_text() for page in pdf_reader.pages[:10]])  # Limit pages
        
    elif "word" in uploaded_file.type:
        doc = docx.Document(uploaded_file)
        return "\n".join([para.text for para in doc.paragraphs[:50]])  # Limit paras
    
    else:
        return uploaded_file.read().decode("utf-8")[:10000]  # Limit size

# Streamlit App
st.set_page_config(page_title="Health Policy AI", layout="wide")

st.title("🏥 Health Policy AI")
st.markdown("**Simplified policy analysis powered by AI**")

# File upload
uploaded_file = st.file_uploader("Upload policy document", type=['pdf', 'docx', 'txt'])

if uploaded_file:
    with st.spinner("Processing..."):
        text = extract_text(uploaded_file)
        st.session_state.text = text[:5000]  # Limit for cloud
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.header("📋 Summary")
        if st.button("Generate Summary"):
            qa_pipe, summary_pipe = load_lightweight_models()
            result = summary_pipe(st.session_state.text[:1000], max_length=150, min_length=30)
            st.success(result[0]['summary'])
    
    with col2:
        st.header("🔍 Q&A")
        question = st.text_input("Ask about the policy:")
        if question and st.button("Answer"):
            qa_pipe, _ = load_lightweight_models()
            result = qa_pipe(question=question, context=st.session_state.text[:2000])
            st.info(result['answer'])
            st.caption(f"Confidence: {result['score']:.1%}")

st.info("💡 Works with PDF, Word, TXT files")
