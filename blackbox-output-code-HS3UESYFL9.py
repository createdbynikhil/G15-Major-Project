import streamlit as st
import PyPDF2
import docx
import pandas as pd
from io import BytesIO
import time
from summarizer import PolicySummarizer
from qa_model import PolicyQA

# Page config
st.set_page_config(
    page_title="Health Policy AI",
    page_icon="🏥",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize models (lazy loading)
@st.cache_resource
def load_models():
    with st.spinner("Loading AI models... This may take a minute."):
        summarizer = PolicySummarizer()
        qa_model = PolicyQA()
    return summarizer, qa_model

class PolicyAnalyzer:
    def __init__(self):
        self.summarizer, self.qa_model = load_models()
    
    def extract_text(self, uploaded_file):
        """Extract text from various file formats"""
        if uploaded_file.type == "application/pdf":
            pdf_reader = PyPDF2.PdfReader(uploaded_file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
            return text
        
        elif uploaded_file.type == "application/vnd.openxmlformats-officedocument.wordprocessingml.document":
            doc = docx.Document(uploaded_file)
            return "\n".join([para.text for para in doc.paragraphs])
        
        else:
            # Plain text
            return uploaded_file.read().decode("utf-8")

# Initialize analyzer
analyzer = PolicyAnalyzer()

# Header
st.title("🏥 Health Policy AI Simplifier")
st.markdown("**Upload complex health policy documents and get instant summaries, key points, and answers to your questions!**")

# Sidebar
st.sidebar.header("Upload Policy Document")
uploaded_file = st.sidebar.file_uploader(
    "Choose a PDF, Word doc, or text file",
    type=['pdf', 'docx', 'txt']
)

# Main content
if uploaded_file is not None:
    with st.spinner("Processing document..."):
        policy_text = analyzer.extract_text(uploaded_file)
        filename = uploaded_file.name
        
        # Store in session state
        if 'policy_text' not in st.session_state:
            st.session_state.policy_text = policy_text
            st.session_state.filename = filename
            st.session_state.generated = False
    
    # Display document info
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Document", st.session_state.filename)
    with col2:
        st.metric("Words", len(st.session_state.policy_text.split()))
    with col3:
        st.metric("Characters", len(st.session_state.policy_text))
    
    # Tabs for different views
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Summary", "🔍 Q&A", "💡 Key Points", "❓ Auto Questions"])
    
    with tab1:
        st.header("📋 Executive Summary")
        if st.button("Generate Summary", key="summary"):
            with st.spinner("Generating summary..."):
                summary = analyzer.summarizer.summarize(st.session_state.policy_text)
                st.session_state.summary = summary
                st.session_state.generated = True
        
        if st.session_state.get('generated', False) and 'summary' in st.session_state:
            st.success("✅ Summary generated!")
            st.markdown("### **Summary**")
            st.write(st.session_state.summary)
    
    with tab2:
        st.header("🔍 Ask Questions About the Policy")
        
        # Question input
        question = st.text_input("What would you like to know?", placeholder="e.g., What are the eligibility criteria?")
        
        if question and st.button("Get Answer", key="qa"):
            with st.spinner("Finding answer..."):
                answer = analyzer.qa_model.answer_question(question, st.session_state.policy_text)
                st.session_state.last_question = question
                st.session_state.last_answer = answer
        
        # Display last answer
        if 'last_answer' in st.session_state:
            st.markdown("### **Answer**")
            st.info(st.session_state.last_answer['answer'])
            st.caption(f"**Confidence:** {st.session_state.last_answer['score']:.1%}")
            with st.expander("📄 Context"):
                st.write(st.session_state.last_answer['context'])
    
    with tab3:
        st.header("💡 Key Points")
        if st.button("Extract Key Points", key="keypoints"):
            with st.spinner("Extracting key points..."):
                keypoints = analyzer.summarizer.generate_key_points(st.session_state.policy_text)
                st.session_state.keypoints = keypoints
        
        if 'keypoints' in st.session_state:
            st.markdown("### **Key Takeaways**")
            st.write(st.session_state.keypoints)
    
    with tab4:
        st.header("❓ Auto-Generated Questions")
        if st.button("Generate Questions", key="questions"):
            with st.spinner("Generating relevant questions..."):
                questions = analyzer.qa_model.generate_questions(st.session_state.policy_text)
                st.session_state.questions = questions[:8]
        
        if 'questions' in st.session_state:
            st.markdown("### **Suggested Questions**")
            for i, q in enumerate(st.session_state.questions, 1):
                if st.button(q, key=f"q{i}"):
                    st.session_state.last_question = q
                    answer = analyzer.qa_model.answer_question(q, st.session_state.policy_text)
                    st.session_state.last_answer = answer
                    st.rerun()

else:
    st.info("👆 **Please upload a policy document to get started!**")
    st.markdown("""
    ### Supported Formats:
    - 📄 PDF files
    - 📝 Microsoft Word (.docx)
    - 📄 Plain text files
    
    ### Example Use Cases:
    - Health insurance policies
    - Medicare/Medicaid guidelines
    - Hospital compliance documents
    - Public health regulations
    """)

# Footer
st.markdown("---")
st.markdown(
    """
    <div style='text-align: center; color: #666;'>
        Built with ❤️ using T5 (summarization) + BERT (Q&A) | 
        <a href='https://huggingface.co'>🤗 Hugging Face</a>
    </div>
    """, 
    unsafe_allow_html=True
)