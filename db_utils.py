import sqlite3
import json
import os
from langchain_community.vectorstores import FAISS
from embedding_generator import get_embeddings

def init_db():
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    
    # Users table for authentication
    c.execute("""
        CREATE TABLE IF NOT EXISTS users (
            username TEXT PRIMARY KEY,
            password TEXT,
            salt TEXT,
            refresh_token TEXT
        )
    """)
    
    # Progress table for quiz scores and flashcard reviews
    c.execute("""
        CREATE TABLE IF NOT EXISTS progress (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            topic TEXT,
            quiz_score INTEGER,
            quiz_total INTEGER,
            flashcards_reviewed INTEGER,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    
    # Quiz history table for detailed quiz data
    c.execute("""
        CREATE TABLE IF NOT EXISTS quiz_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            topic TEXT,
            question TEXT,
            question_type TEXT,
            options TEXT,
            correct_answer TEXT,
            user_answer TEXT,
            is_correct BOOLEAN,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    
    # Chat history table for Q&A interactions
    c.execute("""
        CREATE TABLE IF NOT EXISTS chat_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            question TEXT,
            answer TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    
    # Study plans table for generated study schedules
    c.execute("""
        CREATE TABLE IF NOT EXISTS study_plans (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT,
            subjects TEXT,
            exam_date TEXT,
            interests TEXT,
            study_plan TEXT,
            timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
            FOREIGN KEY (username) REFERENCES users (username)
        )
    """)
    
    conn.commit()
    conn.close()

def log_progress(username, topic, quiz_score=None, quiz_total=None, flashcards_reviewed=None):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO progress (username, topic, quiz_score, quiz_total, flashcards_reviewed)
        VALUES (?, ?, ?, ?, ?)
    """, (username, topic, quiz_score, quiz_total, flashcards_reviewed))
    conn.commit()
    conn.close()

def log_quiz_history(username, topic, question_data, user_answer, is_correct):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO quiz_history (username, topic, question, question_type, options, correct_answer, user_answer, is_correct)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        username,
        topic,
        question_data["question"],
        question_data["type"],
        json.dumps(question_data.get("options", [])),
        question_data["correct_answer"],
        user_answer,
        is_correct
    ))
    conn.commit()
    conn.close()

def log_chat_history(username, question, answer):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO chat_history (username, question, answer)
        VALUES (?, ?, ?)
    """, (username, question, answer))
    conn.commit()
    conn.close()

def log_study_plan(username, subjects, exam_date, interests, study_plan):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("""
        INSERT INTO study_plans (username, subjects, exam_date, interests, study_plan)
        VALUES (?, ?, ?, ?, ?)
    """, (username, subjects, exam_date, interests, json.dumps(study_plan)))
    conn.commit()
    conn.close()

def get_progress(username):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("""
        SELECT topic, quiz_score, quiz_total, flashcards_reviewed, timestamp
        FROM progress
        WHERE username = ?
        ORDER BY timestamp DESC
    """, (username,))
    data = c.fetchall()
    conn.close()
    return data

def get_quiz_history(username):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("""
        SELECT topic, question, question_type, options, correct_answer, user_answer, is_correct, timestamp
        FROM quiz_history
        WHERE username = ?
        ORDER BY timestamp DESC
    """, (username,))
    data = c.fetchall()
    conn.close()
    return data

def get_chat_history(username):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("""
        SELECT question, answer, timestamp
        FROM chat_history
        WHERE username = ?
        ORDER BY timestamp DESC
    """, (username,))
    data = c.fetchall()
    conn.close()
    return data

def get_study_plans(username):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("""
        SELECT subjects, exam_date, interests, study_plan, timestamp
        FROM study_plans
        WHERE username = ?
        ORDER BY timestamp DESC
    """, (username,))
    data = c.fetchall()
    conn.close()
    return data

def save_vector_store(vector_store, username):
    user_vector_path = f"faiss_index_{username}"
    vector_store.save_local(user_vector_path)

def load_vector_store(username):
    user_vector_path = f"faiss_index_{username}"
    if os.path.exists(user_vector_path):
        return FAISS.load_local(user_vector_path, get_embeddings(), allow_dangerous_deserialization=True)
    return None

def delete_user_data(username):
    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()
    c.execute("DELETE FROM users WHERE username = ?", (username,))
    c.execute("DELETE FROM progress WHERE username = ?", (username,))
    c.execute("DELETE FROM quiz_history WHERE username = ?", (username,))
    c.execute("DELETE FROM chat_history WHERE username = ?", (username,))
    c.execute("DELETE FROM study_plans WHERE username = ?", (username,))
    conn.commit()
    conn.close()
    # Delete vector store
    user_vector_path = f"faiss_index_{username}"
    if os.path.exists(user_vector_path):
        import shutil
        shutil.rmtree(user_vector_path)