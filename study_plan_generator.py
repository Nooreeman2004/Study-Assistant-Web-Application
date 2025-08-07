import os
import uuid
import json
import sqlite3
from datetime import datetime
from dotenv import load_dotenv
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from db_utils import log_study_plan
from middleware import protect_page

# Helper Functions
def load_env_variables():
    """Load environment variables."""
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in .env file. Please add it and restart the app.")
        st.stop()
    return groq_api_key

def parse_exam_schedule(exam_dates):
    """Parse exam dates from user input."""
    try:
        return {
            subj.strip(): datetime.strptime(date.strip(), "%Y-%m-%d")
            for line in exam_dates.splitlines() if ":" in line
            for subj, date in [line.split(":")]
        }
    except ValueError:
        st.error("Invalid date format in exam schedules. Use YYYY-MM-DD.")
        return None

def process_uploaded_files(uploaded_files):
    """Load content from uploaded PDF/DOCX files."""
    documents = []
    for file in uploaded_files:
        temp_path = f"temp_{uuid.uuid4()}{os.path.splitext(file.name)[1]}"
        with open(temp_path, "wb") as f:
            f.write(file.getbuffer())
        loader = PyPDFLoader(temp_path) if file.name.endswith(".pdf") else Docx2txtLoader(temp_path)
        documents.extend(loader.load())
        os.remove(temp_path)
    return documents

def extract_topics_from_documents(documents, subjects, embeddings_model):
    """Extract key topics from documents using embeddings and LLM."""
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = HuggingFaceEmbeddings(model_name=embeddings_model, model_kwargs={"device": "cpu"})
    vector_store = FAISS.from_documents(chunks, embeddings)
    st.session_state.vector_store = vector_store

    llm = ChatGroq(groq_api_key=load_env_variables(), model_name="llama3-8b-8192", temperature=0.2)
    context = " ".join([doc.page_content for doc in documents[:5]])
    prompt = PromptTemplate(
        input_variables=["context", "subjects"],
        template="Extract key topics from context for subjects: {subjects}. Context: {context}. Answer:"
    )
    response = llm.invoke(prompt.format(context=context, subjects=subjects))
    return [t.strip() for t in response.content.splitlines() if t.strip()]

def generate_study_plan(timeline_days, daily_hours, subjects, exam_schedule, interests, topics):
    """Generate a personalized study plan using LLM."""
    sorted_exams = sorted(exam_schedule.items(), key=lambda x: x[1])
    all_subjects = [subj for subj, _ in sorted_exams] + [
        s for s in subjects.split(",") if s.strip() not in exam_schedule
    ]
    llm = ChatGroq(groq_api_key=load_env_variables(), model_name="llama3-8b-8192", temperature=0.2)
    plan_prompt = PromptTemplate(
        input_variables=["timeline_days", "daily_hours", "subjects", "exam_schedule", "interests", "topics"],
        template=(
            "Generate a {timeline_days}-day study plan allocating {daily_hours} hours per day. "
            "Subjects: {subjects}. Exam Schedule: {exam_schedule}. Interests: {interests}. "
            "Topics: {topics}. Prioritize earlier exams. Format: Day X: Subject - Topic (Hours)"
        )
    )
    response = llm.invoke(plan_prompt.format(
        timeline_days=timeline_days,
        daily_hours=daily_hours,
        subjects=", ".join(all_subjects),
        exam_schedule=str(exam_schedule),
        interests=interests,
        topics=", ".join(topics)
    ))
    return response.content

def run_study_plan_generator():
    protect_page()  # Validate JWT before rendering page
    groq_api_key = load_env_variables()

    # Initialize session state
    if "study_plan" not in st.session_state:
        st.session_state.study_plan = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # App title and description
    st.title("Personalized Study Plan Generator")
    st.write("Generate a tailored daily study schedule by inputting your subjects, interests, and exam schedules.")

    # User input form
    with st.form("study_plan_form"):
        st.subheader("Enter Your Details")
        subjects = st.text_input("Subjects (comma-separated, e.g., Math, Science)")
        interests = st.text_area("Interests (e.g., algebra, physics)")
        exam_dates = st.text_area("Exam Dates (e.g., Math: 2025-05-10)")
        timeline_days = st.number_input("Preparation Days", min_value=1, max_value=365, value=30)
        daily_hours = st.number_input("Daily Study Hours", min_value=1, max_value=12, value=4)
        uploaded_files = st.file_uploader("Upload Study Materials (PDF/DOCX)", type=["pdf", "docx"], accept_multiple_files=True)
        submit = st.form_submit_button("Generate Plan")

    if submit:
        # Validate and process input
        exam_schedule = parse_exam_schedule(exam_dates)
        if not exam_schedule:
            return
        documents = process_uploaded_files(uploaded_files) if uploaded_files else []
        topics = extract_topics_from_documents(documents, subjects, "sentence-transformers/all-MiniLM-L6-v2") if documents else []

        # Generate and display study plan
        study_plan = generate_study_plan(timeline_days, daily_hours, subjects, exam_schedule, interests, topics)
        st.session_state.study_plan = study_plan
        log_study_plan(st.session_state.user_id, subjects, exam_schedule, interests, study_plan)
        st.success("Study plan generated!")

    # Display study plan
    if st.session_state.study_plan:
        st.subheader("Your Study Plan")
        st.text(st.session_state.study_plan)