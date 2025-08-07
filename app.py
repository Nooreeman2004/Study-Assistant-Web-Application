import streamlit as st
import sqlite3
import hashlib
import os
import binascii
from real_time_qa import run_study_assistant
from study_plan_generator import run_study_plan_generator
from quiz_generator import run_quiz_generator
from flashcards import run_flashcards_creator
from progress_tracker import run_progress_tracker
from study_suggestion import run_study_suggestion
from dashboard import run_dashboard
from db_utils import init_db, log_progress, delete_user_data

# Set page config as the first Streamlit command
st.set_page_config(page_title="Study Assistant", page_icon="📚", initial_sidebar_state="expanded")

# Global CSS for black background and consistent styling
st.markdown("""
    <style>
    .stApp {
        background-color: #000000;
        color: #FFFFFF;
    }
    h1, h2, h3 {
        color: #FF69B4;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stButton>button {
        background-color: #FF69B4;
        color: #000000;
        border-radius: 10px;
        font-family: 'Comic Sans MS', cursive, sans-serif;
    }
    .stTextInput>div>input {
        background-color: #333333;
        color: #FFFFFF;
        border-radius: 10px;
    }
    .stRadio>div {
        color: #FFFFFF;
    }
    .stFileUploader label {
        color: #FFFFFF;
    }
    .stSuccess {
        background-color: #333333;
        color: #FF69B4;
    }
    .stError {
        background-color: #333333;
        color: #FF0000;
    }
    </style>
""", unsafe_allow_html=True)

# Initialize database
init_db()

# Session state initialization
if "current_page" not in st.session_state:
    st.session_state["current_page"] = "Login"
if "user_id" not in st.session_state:
    st.session_state["user_id"] = None
if "reset_password" not in st.session_state:
    st.session_state["reset_password"] = False

def hash_password(password, salt_hex):
    salt = binascii.unhexlify(salt_hex)
    salted = password.encode() + salt
    return hashlib.sha256(salted).hexdigest()

def signup():
    st.header("Sign Up")
    username = st.text_input("Choose a username:", key="signup_username")
    password = st.text_input("Choose a password:", type="password", key="signup_password")
    confirm_password = st.text_input("Confirm password:", type="password", key="signup_confirm_password")
    
    if st.button("Sign Up"):
        if not username or not password or not confirm_password:
            st.error("Please fill in all fields.")
        elif password != confirm_password:
            st.error("Passwords do not match.")
        else:
            try:
                conn = sqlite3.connect("study_assistant.db")
                c = conn.cursor()
                c.execute("SELECT username FROM users WHERE username = ?", (username,))
                if c.fetchone():
                    st.error("Username already exists.")
                else:
                    salt = os.urandom(32)
                    salt_hex = salt.hex()
                    hashed_password = hash_password(password, salt_hex)
                    c.execute("INSERT INTO users (username, password, salt) VALUES (?, ?, ?)", 
                              (username, hashed_password, salt_hex))
                    conn.commit()
                    st.success(f"Account created for {username}! Please log in. 😺")
                    st.session_state["current_page"] = "Login"
                    st.rerun()
                conn.close()
            except Exception as e:
                st.error(f"Signup failed: {e}")

def reset_password():
    st.header("Reset Password")
    username = st.text_input("Enter your username:", key="reset_username")
    new_password = st.text_input("Enter new password:", type="password", key="reset_new_password")
    confirm_password = st.text_input("Confirm new password:", type="password", key="reset_confirm_password")
    
    if st.button("Reset Password"):
        if not username or not new_password or not confirm_password:
            st.error("Please fill in all fields.")
        elif new_password != confirm_password:
            st.error("Passwords do not match.")
        else:
            try:
                conn = sqlite3.connect("study_assistant.db")
                c = conn.cursor()
                c.execute("SELECT username FROM users WHERE username = ?", (username,))
                if not c.fetchone():
                    st.error("Username not found.")
                else:
                    salt = os.urandom(32)
                    salt_hex = salt.hex()
                    hashed_password = hash_password(new_password, salt_hex)
                    c.execute("UPDATE users SET password = ?, salt = ? WHERE username = ?", 
                              (hashed_password, salt_hex, username))
                    conn.commit()
                    st.success(f"Password reset for {username}! Please log in. 😺")
                    st.session_state["reset_password"] = False
                    st.session_state["current_page"] = "Login"
                    st.rerun()
                conn.close()
            except Exception as e:
                st.error(f"Reset failed: {e}")
    
    if st.button("Back to Login"):
        st.session_state["reset_password"] = False
        st.session_state["current_page"] = "Login"
        st.rerun()

def login():
    st.header("Log In")
    username = st.text_input("Enter your username:", key="login_username")
    password = st.text_input("Enter your password:", type="password", key="login_password")
    
    if st.button("Log In"):
        try:
            conn = sqlite3.connect("study_assistant.db")
            c = conn.cursor()
            c.execute("SELECT password, salt FROM users WHERE username = ?", (username,))
            result = c.fetchone()
            if result:
                stored_password, salt_hex = result
                hashed_input_password = hash_password(password, salt_hex)
                if hashed_input_password == stored_password:
                    # Clear previous session state
                    for key in list(st.session_state.keys()):
                        if key not in ["current_page", "user_id", "reset_password"]:
                            del st.session_state[key]
                    st.session_state["user_id"] = username
                    st.session_state["current_page"] = "Dashboard"
                    st.success(f"Welcome back, {username}! 🌟")
                    st.rerun()
                else:
                    st.error("Invalid username or password")
            else:
                st.error("Invalid username or password")
            conn.close()
        except Exception as e:
            st.error(f"Login failed: {e}")
    
    if st.button("Forgot Password?"):
        st.session_state["reset_password"] = True
        st.session_state["current_page"] = "Login"
        st.rerun()

def main():
    if st.session_state["current_page"] == "Login":
        if st.session_state.get("reset_password", False):
            reset_password()
        else:
            login()
            st.write("Don't have an account? Go to Sign Up!")
            if st.button("Go to Sign Up"):
                st.session_state["current_page"] = "Signup"
                st.rerun()
    elif st.session_state["current_page"] == "Signup":
        signup()
        st.write("Already have an account? Go to Log In!")
        if st.button("Go to Log In"):
            st.session_state["current_page"] = "Login"
            st.rerun()
    elif st.session_state["current_page"] == "Dashboard":
        run_dashboard()

if __name__ == "__main__":
    main()