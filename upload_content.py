import streamlit as st
import os
import uuid
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage, HumanMessagePromptTemplate
from langchain.chains import LLMChain
from langchain.memory import ConversationBufferWindowMemory
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.document_loaders import PyPDFLoader
from langchain_core.output_parsers import JsonOutputParser
import tempfile
from middleware import protect_page

# Load environment variables
load_dotenv()
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize session state
def init_session_state():
    defaults = {
        "vectorstore": None,
        "chat_history": [],
        "thread_id": str(uuid.uuid4()),
        "system_prompt": "You are a helpful study assistant. Use the uploaded content to answer questions, provide summaries, explanations, or generate quizzes. Be clear, concise, and educational.",
        "model": "llama3-8b-8192",
        "memory_length": 5
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value

# Initialize Groq LLM
def init_llm():
    try:
        return ChatGroq(
            groq_api_key=groq_api_key,
            model_name=st.session_state["model"],
            temperature=0.7
        )
    except Exception as e:
        st.error(f"Error initializing Groq LLM: {e}")
        return None

# Initialize embeddings and text splitter
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=100)

# Chatbot function
def real_time_qa():
    protect_page()  # Validate JWT before rendering page
    init_session_state()
    st.title("Study Buddy Chatbot")
    st.write("Upload a PDF study material and ask questions, request summaries, explanations, or quizzes!")

    # Sidebar for customization
    with st.sidebar:
        st.title("Customization")
        st.session_state["system_prompt"] = st.text_input(
            "System Prompt:",
            value=st.session_state["system_prompt"]
        )
        st.session_state["model"] = st.selectbox(
            "Choose a model",
            ["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"],
            index=["llama3-8b-8192", "mixtral-8x7b-32768", "gemma-7b-it"].index(st.session_state["model"])
        )
        st.session_state["memory_length"] = st.slider("Conversational Memory Length:", 1, 10, value=st.session_state["memory_length"])

    # Initialize memory
    memory = ConversationBufferWindowMemory(
        k=st.session_state["memory_length"],
        memory_key="chat_history",
        return_messages=True
    )

    # Initialize LLM
    groq_chat = init_llm()
    if not groq_chat:
        return

    # Chat prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessage(content=st.session_state["system_prompt"]),
        MessagesPlaceholder(variable_name="chat_history"),
        HumanMessagePromptTemplate.from_template("{human_input}")
    ])

    # Conversation chain
    conversation = LLMChain(
        llm=groq_chat,
        prompt=prompt,
        memory=memory,
        verbose=False
    )

    # Quiz generation prompt
    quiz_parser = JsonOutputParser()
    quiz_prompt = ChatPromptTemplate.from_messages([
        ("system", """Generate 4 multiple-choice questions based on the provided content. Each question should have 4 options and one correct answer. Output in JSON format:
        [
            {
                "question": "Question text",
                "options": ["Option 1", "Option 2", "Option 3", "Option 4"],
                "correct_answer": "Correct option"
            },
            ...
        ]
        Content: {content}"""),
        ("user", "Generate the quiz.")
    ])
    quiz_chain = quiz_prompt | groq_chat | quiz_parser

    # File upload
    uploaded_file = st.file_uploader("Upload a PDF study material", type=["pdf"], key=f"uploader_{st.session_state['thread_id']}")
    if uploaded_file:
        try:
            # Save uploaded file to a temporary location
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getbuffer())
                tmp_file_path = tmp_file.name

            # Load and process PDF
            loader = PyPDFLoader(tmp_file_path)
            documents = loader.load()
            texts = text_splitter.split_documents(documents)

            # Create vector store
            st.session_state.vectorstore = Chroma.from_documents(
                documents=texts,
                embedding=embeddings,
                collection_name=f"study_material_{st.session_state['thread_id']}"
            )
            st.success("PDF uploaded and processed successfully!")

            # Clean up temporary file
            os.unlink(tmp_file_path)
        except Exception as e:
            st.error(f"Error processing PDF: {e}")

    # User input
    user_question = st.text_input("Ask a question, or request a summary, explanation, or quiz:", key=f"input_{st.session_state['thread_id']}")
    if st.button("Submit") and user_question.strip():
        try:
            # Load chat history into memory
            for message in st.session_state.chat_history:
                memory.save_context(
                    {"input": message["human"]},
                    {"output": message["AI"]}
                )

            # Handle different request types
            if "quiz" in user_question.lower() and st.session_state.vectorstore:
                docs = st.session_state.vectorstore.similarity_search(user_question, k=2)
                content = " ".join([doc.page_content for doc in docs])
                quiz_result = quiz_chain.p ({"content": content})
                response = "Here is your quiz:\n
                