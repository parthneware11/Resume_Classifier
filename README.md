
📄 AI Resume Classification System

An end-to-end Machine Learning + NLP project that classifies resumes into suitable job roles using a Streamlit web application.

This project demonstrates skills in text preprocessing, TF-IDF vectorization, ML modeling, and deployment.

🚀 Features

📂 Upload resumes (PDF / DOCX)

🧠 NLP-based text cleaning

📊 Job role prediction with confidence score

⚡ Fast & interactive Streamlit UI

💾 Models loaded using Pickle (.pkl)

🧠 Tech Stack

Python

Scikit-learn

TF-IDF Vectorizer

NLTK

Streamlit

Pickle (.pkl)

Git & GitHub

📁 Project Structure
Resume Classifier/
│
├── resume_app.py        # Streamlit application
├── clf.pkl              # Trained ML model
├── tfidf.pkl            # TF-IDF vectorizer
├── requirements.txt     # Python dependencies
├── README.md            # Project documentation

⚙️ How It Works

User uploads a resume (PDF or DOCX)

Resume text is extracted

Text is cleaned using NLP techniques

TF-IDF converts text to numerical features

ML model predicts the job category

App displays prediction + confidence score