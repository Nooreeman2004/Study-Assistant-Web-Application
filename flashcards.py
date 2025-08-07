import streamlit as st
import os
import uuid
import json
import re
import sqlite3
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer
import torch
from middleware import protect_page

def run_flashcards_creator():
    protect_page()  # Validate JWT before rendering page
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in .env file. Please add it and restart the app.")
        st.stop()

    # Initialize session state
    if "flashcards" not in st.session_state:
        st.session_state.flashcards = None
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "flashcard_index" not in st.session_state:
        st.session_state.flashcard_index = 0

    # Streamlit app title and description
    st.title("Flashcards Creator")
    st.write("Generate flashcards from your study materials by uploading PDFs or Word documents.")

    # User input form for flashcard generation
    with st.form("flashcards_form"):
        uploaded_files = st.file_uploader(
            "Upload Study Materials (PDF or DOCX)",
            type=["pdf", "docx"],
            accept_multiple_files=True,
            disabled=False
        )
        if uploaded_files:
            st.write(f"Uploaded files: {[file.name for file in uploaded_files]}")

        num_flashcards = st.number_input("Number of Flashcards", min_value=1, max_value=20, value=5)
        topic = st.text_input("Topic (optional, e.g., Calculus, Python)", key="flashcard_topic")
        submit_button = st.form_submit_button("Generate Flashcards")

    # Function to process uploaded files
    def process_files(files):
        documents = []
        for file in files:
            file_extension = os.path.splitext(file.name)[1].lower()
            temp_file_path = f"temp_{uuid.uuid4()}{file_extension}"
            with open(temp_file_path, "wb") as f:
                f.write(file.getbuffer())

            if file_extension == ".pdf":
                loader = PyPDFLoader(temp_file_path)
                documents.extend(loader.load())
            elif file_extension == ".docx":
                loader = Docx2txtLoader(temp_file_path)
                documents.extend(loader.load())

            os.remove(temp_file_path)

        return documents

    # Custom embedding function
    class CustomEmbeddings:
        def __init__(self, model_name):
            self.model = SentenceTransformer(model_name, device="cpu")

        def embed_documents(self, texts):
            return self.model.encode(texts, show_progress_bar=False).tolist()

        def embed_query(self, text):
            return self.model.encode([text], show_progress_bar=False)[0].tolist()

        def __call__(self, texts):
            return self.embed_documents(texts)

    # Function to create or reuse vector store
    def get_vector_store(documents):
        if documents:
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
            chunks = text_splitter.split_documents(documents)
            embeddings = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            return FAISS.from_documents(chunks, embeddings)
        return st.session_state.vector_store

    # Function to log progress
    def log_progress(username, topic, quiz_score=None, quiz_total=None, flashcards_reviewed=None):
        conn = sqlite3.connect("study_assistant.db")
        c = conn.cursor()
        c.execute("""
            INSERT INTO progress (username, topic, quiz_score, quiz_total, flashcards_reviewed)
            VALUES (?, ?, ?, ?, ?)
        """, (username, topic, quiz_score, quiz_total, flashcards_reviewed))
        conn.commit()
        conn.close()

    # Function to generate flashcards
    def generate_flashcards(content, num_flashcards):
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0.3)
        prompt_template = """
        You are an expert flashcard generator. Your task is to create exactly {num_flashcards} flashcards based on the provided content. Follow these instructions strictly:

        - Base the flashcards on this content: {content}
        - Each flashcard should consist of a question and a concise answer (1-2 sentences).
        - The question should test understanding of a key concept, definition, or fact from the content.
        - The answer should be accurate and directly address the question.

        Output the flashcards in strict JSON format with the following structure. Do not include any additional text, comments, or explanations outside the JSON:
        [
            {{
                "question": "Question text",
                "answer": "Answer text"
            }},
            ...
        ]

        Ensure the JSON is valid and contains exactly {num_flashcards} flashcards. Start your response with `[` and end with `]`.
        """
        prompt = PromptTemplate(
            input_variables=["num_flashcards", "content"],
            template=prompt_template
        )

        # Prepare content with LangChain's FAISS retriever
        if content and st.session_state.vector_store:
            retriever = st.session_state.vector_store.as_retriever()
            relevant_docs = retriever.get_relevant_documents("general")
            content = " ".join([doc.page_content for doc in relevant_docs])
        else:
            content = "General knowledge."

        # Ensure content is not empty
        if not content.strip():
            content = "General knowledge."

        # Generate flashcards
        response = llm.invoke(prompt.format(
            num_flashcards=num_flashcards,
            content=content[:5000]  # Limit content length
        ))

        # Debug: Print the raw response
        st.write("Debug - Raw LLM Response:", response.content)

        # Extract the JSON portion of the response
        json_match = re.search(r'\[.*\]', response.content, re.DOTALL)
        if not json_match:
            st.error("Failed to find JSON in the response. Please try again.")
            return None

        json_str = json_match.group(0)

        # Try to parse the extracted JSON
        try:
            flashcards = json.loads(json_str)
            # Validate that the flashcards have the correct number
            if len(flashcards) != num_flashcards:
                st.error(f"Expected {num_flashcards} flashcards, but got {len(flashcards)}. Please try again.")
                return None
            # Validate that each flashcard has the required fields
            for f in flashcards:
                if "question" not in f or "answer" not in f:
                    st.error("Invalid flashcard format: Missing required fields.")
                    return None
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse extracted JSON: {str(e)}")
            return None

        return flashcards

    # Process form submission for flashcard generation
    if submit_button:
        with st.spinner("Generating flashcards..."):
            documents = None

            # Process any uploaded files
            if uploaded_files:
                documents = process_files(uploaded_files)
                if documents:
                    if st.session_state.vector_store:
                        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
                        chunks = text_splitter.split_documents(documents)
                        embeddings = CustomEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
                        new_vector_store = FAISS.from_documents(chunks, embeddings)
                        st.session_state.vector_store.merge_from(new_vector_store)
                    else:
                        st.session_state.vector_store = get_vector_store(documents)
                st.write(f"Debug - Vector store updated: {st.session_state.vector_store is not None}")

            # Check if vector store exists
            if not st.session_state.vector_store:
                st.error("Please upload study materials to generate flashcards.")
                return

            # Generate flashcards
            flashcards = generate_flashcards(documents, num_flashcards)
            if flashcards:
                st.session_state.flashcards = flashcards
                st.session_state.flashcard_index = 0  # Reset index
                st.success("Flashcards generated successfully!")

                # Log topic completion
                username = st.session_state.get("user_id")
                if username and topic:
                    log_progress(username, topic)

    # Interactive flashcard review
    if st.session_state.flashcards:
        st.subheader("Review Flashcards")
        num_flashcards = len(st.session_state.flashcards)
        current_index = st.session_state.flashcard_index

        if num_flashcards > 0:
            card = st.session_state.flashcards[current_index]
            st.write(f"**Flashcard {current_index + 1} of {num_flashcards}**")
            with st.expander("Show Question"):
                st.write(f"**Question**: {card['question']}")
                if st.button("Reveal Answer", key=f"reveal_{current_index}"):
                    st.write(f"**Answer**: {card['answer']}")

            # Navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if current_index > 0:
                    if st.button("Previous", key="prev_flashcard"):
                        st.session_state.flashcard_index -= 1
                        st.rerun()
            with col2:
                if current_index < num_flashcards - 1:
                    if st.button("Next", key="next_flashcard"):
                        st.session_state.flashcard_index += 1
                        st.session_state.flashcard_index = min(st.session_state.flashcard_index, num_flashcards - 1)
                        # Log flashcard review when moving to the next card
                        username = st.session_state.get("user_id")
                        if username and topic:
                            log_progress(username, topic, flashcards_reviewed=1)
                        st.rerun()