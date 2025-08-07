import streamlit as st
import os
import sqlite3
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate
from langchain_groq import ChatGroq
from langchain.chains import ConversationalRetrievalChain
from embedding_generator import get_embeddings
from db_utils import log_chat_history
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

def create_conversation_chain(documents):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_documents(documents)
    embeddings = get_embeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in environment variables.")
        return None, None
    
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0.2
    )
    
    prompt_template = """
    You are a helpful study assistant. Use the provided context to answer questions or summarize content accurately and concisely. If the context contains relevant information, base your answer on it. If the context is empty or irrelevant, use your general knowledge to provide an accurate answer, and state: "Based on general knowledge, [your answer]." If you are unsure or don't know the answer, say so.

    Context: {context}
    Question: {question}
    Chat History: {chat_history}

    Answer:
    """
    prompt = PromptTemplate(
        input_variables=["context", "question", "chat_history"],
        template=prompt_template
    )
    
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vector_store.as_retriever(),
        combine_docs_chain_kwargs={"prompt": prompt}
    )
    
    return vector_store, conversation_chain

def run_study_assistant():
    protect_page()  # Validate JWT before rendering page
    st.header("Study Assistant (Q&A) 💬")
    st.write("Ask questions about your study materials or anything else! 📚😺")

    # Initialize session state
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []
    if "vector_store" not in st.session_state:
        st.session_state.vector_store = None
    if "conversation_chain" not in st.session_state:
        st.session_state.conversation_chain = None

    # File upload
    uploaded_files = st.file_uploader("Upload PDF or Word documents:", type=["pdf", "docx"], accept_multiple_files=True)
    if uploaded_files:
        with st.spinner("Processing uploaded files..."):
            documents = process_files(uploaded_files)
            vector_store, conversation_chain = create_conversation_chain(documents)
            st.session_state.vector_store = vector_store
            st.session_state.conversation_chain = conversation_chain
            st.success("Documents processed successfully! 🎉")

    # Question type toggle
    question_type = st.radio("Question Type", ["Document-Based", "General Knowledge"])

    # Chat interface
    st.subheader("Ask a Question")
    user_input = st.text_input("Enter your question here:", key="qa_input")
    if st.button("Submit Question"):
        if not user_input:
            st.error("Please enter a question.")
        elif question_type == "Document-Based" and not st.session_state.conversation_chain:
            st.error("Please upload documents for document-based questions.")
        else:
            with st.spinner("Generating response..."):
                if question_type == "Document-Based":
                    # Use RAG pipeline
                    response = st.session_state.conversation_chain({
                        "question": user_input,
                        "chat_history": st.session_state.chat_history
                    })
                    answer = response["answer"]
                else:
                    # Direct LLM query for general knowledge
                    groq_api_key = os.getenv("GROQ_API_KEY")
                    llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0.2)
                    prompt = f"You are a helpful study assistant. Answer the following question using your general knowledge: {user_input}"
                    response = llm.invoke(prompt)
                    answer = f"Based on general knowledge, {response.content}"

                # Save to chat history
                st.session_state.chat_history.append((user_input, answer))
                
                # Log to database
                if st.session_state.user_id:
                    log_chat_history(st.session_state.user_id, user_input, answer)

                st.write("**Answer:**")
                st.write(answer)

    # Display chat history
    if st.session_state.chat_history:
        st.subheader("Chat History")
        for i, (q, a) in enumerate(st.session_state.chat_history):
            with st.expander(f"Question {i+1}: {q}"):
                st.write(f"**Answer**: {a}")