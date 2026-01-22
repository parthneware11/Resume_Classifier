import streamlit as st
import joblib
import re
import nltk
import numpy as np
import pdfplumber
from docx import Document

from nltk.corpus import stopwords

# ---------------------------------
# App Configuration
# ---------------------------------
st.set_page_config(
    page_title="Resume Classifier",
    page_icon="📄",
    layout="wide"
)

nltk.download("stopwords", quiet=True)

# ---------------------------------
# Load Model
# ---------------------------------
@st.cache_resource
def load_models():
    model = joblib.load("clf.joblib")
    vectorizer = joblib.load("tfidf.joblib")
    return model, vectorizer

model, vectorizer = load_models()

# ---------------------------------
# Text Cleaning
# ---------------------------------
def clean_text(text):
    text = re.sub(r"http\S+", " ", text)
    text = re.sub(r"[^a-zA-Z]", " ", text)
    text = text.lower()
    words = text.split()
    words = [w for w in words if w not in stopwords.words("english")]
    return " ".join(words)

# ---------------------------------
# Resume Text Extraction
# ---------------------------------
def extract_text_from_pdf(file):
    text = ""
    with pdfplumber.open(file) as pdf:
        for page in pdf.pages:
            if page.extract_text():
                text += page.extract_text()
    return text

def extract_text_from_docx(file):
    doc = Document(file)
    return " ".join([para.text for para in doc.paragraphs])

# ---------------------------------
# Custom CSS
# ---------------------------------
st.markdown("""
<style>
.main {
    background: linear-gradient(to right, #0f2027, #203a43, #2c5364);
}
h1, h2, h3 {
    color: #00ffd5;
}
.card {
    background-color: #1e1e1e;
    padding: 20px;
    border-radius: 15px;
    box-shadow: 0px 0px 15px rgba(0,255,213,0.2);
    color: white;
}
.footer {
    text-align: center;
    color: gray;
    margin-top: 40px;
}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("📌 Navigation")
page = st.sidebar.radio(
    "Go to",
    ["🏠 Home", "📤 Upload Resume", "ℹ️ About"]
)

# ---------------------------------
# HOME PAGE
# ---------------------------------
if page == "🏠 Home":
    st.markdown("<h1>📄 Resume Classification System</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <h3>🔍 What does this app do?</h3>
        <p>
        Upload your <b>resume file</b> and the system will automatically
        classify it into the most relevant job role using
        <b>Machine Learning & NLP</b>.
        </p>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("### ⚙️ How it works")
    st.write("""
    1️⃣ Upload resume (PDF / DOCX)  
    2️⃣ Resume text is extracted  
    3️⃣ Text cleaned & vectorized  
    4️⃣ ML model predicts category 🎯
    """)

# ---------------------------------
# UPLOAD & CLASSIFY PAGE
# ---------------------------------
elif page == "📤 Upload Resume":
    st.markdown("<h1>📤 Upload Your Resume</h1>", unsafe_allow_html=True)

    uploaded_file = st.file_uploader(
        "📄 Upload Resume (PDF or DOCX)",
        type=["pdf", "docx"]
    )

    if uploaded_file is not None:
        with st.spinner("📑 Extracting resume text..."):
            if uploaded_file.type == "application/pdf":
                resume_text = extract_text_from_pdf(uploaded_file)
            else:
                resume_text = extract_text_from_docx(uploaded_file)

        if st.button("🚀 Classify Resume"):
            if resume_text.strip() == "":
                st.error("❌ Could not extract text from resume.")
            else:
                cleaned_text = clean_text(resume_text)
                vectorized_text = vectorizer.transform([cleaned_text])

                prediction = model.predict(vectorized_text)[0]
                confidence = np.max(model.predict_proba(vectorized_text)) * 100

                st.success("✅ Resume Classified Successfully")

                st.markdown("<div class='card'>", unsafe_allow_html=True)
                st.markdown(f"### 🎯 Predicted Category: **{prediction}**")
                st.progress(int(confidence))
                st.write(f"📊 Confidence Score: **{confidence:.2f}%**")
                st.markdown("</div>", unsafe_allow_html=True)

# ---------------------------------
# ABOUT PAGE
# ---------------------------------
else:
    st.markdown("<h1>ℹ️ About This Project</h1>", unsafe_allow_html=True)

    st.markdown("""
    <div class="card">
        <ul>
            <li><b>Model:</b> Logistic Regression</li>
            <li><b>Vectorization:</b> TF-IDF</li>
            <li><b>Input:</b> PDF / DOCX resumes</li>
            <li><b>Frontend:</b> Streamlit</li>
        </ul>
        <p>
        Designed to automate resume screening and
        reduce manual effort in recruitment.
        </p>
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------
# Footer
# ---------------------------------
st.markdown("""
<div class="footer">
<hr>
<p>🚀 Built with ❤️ using Streamlit | Resume Classification</p>
</div>
""", unsafe_allow_html=True)
 