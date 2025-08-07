import streamlit as st
import os
import json
import re
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
from embedding_generator import get_embeddings
from db_utils import log_quiz_history, log_progress, load_vector_store, save_vector_store
from middleware import protect_page

def process_files(uploaded_files):
    documents = []
    for file in uploaded_files:
        file_path = f"temp_{file.name}"
        with open(file_path, "wb") as f:
            f.write(file.read())
        if file.name.endswith(".pdf"):
            loader = PyPDFLoader(file_path)
            documents.extend(loader.load())
        elif file.name.endswith(".docx"):
            loader = Docx2txtLoader(file_path)
            documents.extend(loader.load())
        os.remove(file_path)
    return documents

def create_vector_store(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    return vector_store

def extract_topics(vector_store):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables.")
        return "General knowledge"

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0.2
    )

    retriever = vector_store.as_retriever()
    context = retriever.get_relevant_documents("general")
    context_text = "\n".join([doc.page_content for doc in context])

    prompt = PromptTemplate(
        input_variables=["context"],
        template="""
        You are a study assistant. Based on the provided context, identify the main topic or subject covered. Return a single concise topic name (e.g., "Artificial Intelligence", "Calculus"). Do not include additional text or explanations.

        Context: {context}
        """
    )

    try:
        response = llm.invoke(prompt.format(context=context_text))
        return response.content.strip()
    except Exception as e:
        st.error(f"Failed to extract topic: {e}")
        return "General knowledge"

def generate_quiz_questions(vector_store, topic, num_questions=5):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables.")
        return []

    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0.2
    )

    retriever = vector_store.as_retriever() if vector_store else None
    context_text = "\n".join([doc.page_content for doc in retriever.get_relevant_documents(topic or "general")]) if retriever else "No specific context provided."

    prompt = f"""
    You are a helpful study assistant. Generate {num_questions} unique multiple-choice quiz questions about the topic "{topic or 'General knowledge'}". Follow these rules strictly:

    - Return only a valid JSON list of objects, with no additional text, comments, or notes before or after the JSON.
    - Each object must have the following structure:
      {{
        "question": "The question text",
        "type": "multiple_choice",
        "options": ["option 1", "option 2", "option 3", "option 4"],
        "correct_answer": "option X"  // Must be the full text of the correct option
      }}
    - The `correct_answer` must match one of the options exactly (e.g., "To help students excel in their studies", not "b").
    - Do not use letters (e.g., "a", "b", "c", "d") for `correct_answer`.
    - Ensure all questions are unique and relevant to the topic.
    - Base questions on the provided context. If the context is insufficient, use general knowledge and note in the question text: "Based on general knowledge, ..."

    Context: {context_text}

    Start your response with `[` and end with `]`.
    """

    try:
        response = llm.invoke(prompt)
        raw_output = response.content.strip()
        
        # Log the raw response for debugging
        st.write("Debug: Raw response from Groq:", raw_output)

        # Extract JSON using regex to handle extraneous text
        json_match = re.search(r'\[.*\]', raw_output, re.DOTALL)
        if not json_match:
            st.error("Failed to extract JSON from response.")
            return []

        json_str = json_match.group(0)

        # Parse the extracted JSON
        try:
            quiz_questions = json.loads(json_str)
            # Validate the structure
            if not isinstance(quiz_questions, list):
                st.error("Failed to generate quiz: Response is not a JSON list.")
                return []
            
            # Fallback for letter-based correct_answer
            letter_to_index = {'a': 0, 'b': 1, 'c': 2, 'd': 3}
            for q in quiz_questions:
                if not all(key in q for key in ["question", "type", "options", "correct_answer"]):
                    st.error("Failed to generate quiz: Invalid question format.")
                    return []
                # Check if correct_answer is a letter
                if q["correct_answer"].lower() in letter_to_index:
                    index = letter_to_index[q["correct_answer"].lower()]
                    if index < len(q["options"]):
                        q["correct_answer"] = q["options"][index]
                    else:
                        st.error("Failed to generate quiz: Invalid letter-based correct answer.")
                        return []
                # Verify correct_answer is in options
                if q["correct_answer"] not in q["options"]:
                    st.error("Failed to generate quiz: Correct answer does not match any option.")
                    return []
            return quiz_questions
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse quiz questions: {e}")
            st.write("Debug: Invalid JSON extracted:", json_str)
            return []
    except Exception as e:
        st.error(f"Failed to generate quiz: {e}")
        return []

def run_quiz_generator():
    protect_page()  # Validate JWT before rendering page
    st.header("Quiz Generator 📝")
    st.write("Generate a quiz from your study materials or a topic! 😺")

    # Initialize session state
    if "quiz_questions" not in st.session_state:
        st.session_state.quiz_questions = []
    if "quiz_submitted" not in st.session_state:
        st.session_state.quiz_submitted = False
    if "user_answers" not in st.session_state:
        st.session_state.user_answers = []
    if "quiz_topic" not in st.session_state:
        st.session_state.quiz_topic = ""
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None

    # Load vector store if it exists
    if st.session_state.user_id and not st.session_state.vector_store:
        st.session_state.vector_store = load_vector_store(st.session_state.user_id)

    # File upload
    uploaded_files = st.file_uploader("Upload PDF or Word documents (optional):", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            documents = process_files(uploaded_files)
            vector_store = create_vector_store(documents)
            st.session_state.vector_store = vector_store
            save_vector_store(vector_store, st.session_state.user_id)
            st.success("Documents processed successfully! 🎉")
            # Infer topic from documents
            st.session_state.quiz_topic = extract_topics(st.session_state.vector_store)
            st.info(f"Topic inferred from documents: {st.session_state.quiz_topic}")

    # Topic input (only if no documents uploaded)
    topic = ""
    if not uploaded_files:
        topic = st.text_input("Enter the quiz topic:", key="quiz_topic_input")
        st.session_state.quiz_topic = topic or "General knowledge"

    # Number of questions
    num_questions = st.slider("Number of questions:", 1, 10, 5)

    if st.button("Generate Quiz"):
        if not st.session_state.vector_store and not topic:
            st.error("Please upload documents or enter a quiz topic.")
        else:
            with st.spinner("Generating quiz questions..."):
                st.session_state.quiz_questions = generate_quiz_questions(
                    st.session_state.vector_store, st.session_state.quiz_topic, num_questions
                )
                st.session_state.quiz_submitted = False
                st.session_state.user_answers = []
                if st.session_state.quiz_questions:
                    st.success("Quiz generated successfully! 🎉")
                else:
                    st.error("Failed to generate quiz questions.")

    # Display quiz
    if st.session_state.quiz_questions and not st.session_state.quiz_submitted:
        st.subheader(f"Quiz: {st.session_state.quiz_topic}")
        with st.form(key="quiz_form"):
            answers = []
            for i, question_data in enumerate(st.session_state.quiz_questions):
                st.write(f"**Question {i+1}:** {question_data['question']}")
                answer = st.radio(
                    f"Choose an answer for question {i+1}:",
                    question_data["options"],
                    key=f"q_{i}"
                )
                answers.append(answer)
            submit_button = st.form_submit_button("Submit Quiz")

            if submit_button:
                st.session_state.quiz_submitted = True
                st.session_state.user_answers = answers
                st.rerun()

    # Display results
    if st.session_state.quiz_submitted and st.session_state.quiz_questions:
        st.subheader("Quiz Results 🏆")
        score = 0
        total_questions = len(st.session_state.quiz_questions)
        
        for i, (question_data, user_answer) in enumerate(zip(st.session_state.quiz_questions, st.session_state.user_answers)):
            correct_answer = question_data["correct_answer"]
            is_correct = user_answer == correct_answer
            if is_correct:
                score += 1
            with st.expander(f"Question {i+1}: {question_data['question']}"):
                st.write(f"Your answer: {user_answer}")
                st.write(f"Correct answer: {correct_answer}")
                st.write("Result: " + ("Correct 🎉" if is_correct else "Incorrect 😿"))
            # Log to database
            log_quiz_history(st.session_state.user_id, st.session_state.quiz_topic, question_data, user_answer, is_correct)

        st.write(f"Your score: {score} out of {total_questions} ({score/total_questions*100:.1f}%)")
        # Log progress
        log_progress(st.session_state.user_id, st.session_state.quiz_topic, score, total_questions, None)

        if st.button("Start New Quiz"):
            st.session_state.quiz_questions = []
            st.session_state.quiz_submitted = False
            st.session_state.user_answers = []
            st.session_state.quiz_topic = ""
            st.session_state.vector_store = None
            st.rerun()