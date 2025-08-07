import streamlit as st
from real_time_qa import run_study_assistant
from study_plan_generator import run_study_plan_generator
from quiz_generator import run_quiz_generator
from flashcards import run_flashcards_creator
from progress_tracker import run_progress_tracker
from study_suggestion import run_study_suggestion
from db_utils import get_progress, get_quiz_history
from middleware import protect_page

def run_dashboard():
    protect_page()  # Validate JWT before rendering dashboard
    st.header("Study Assistant Dashboard 🌟")
    st.write("Welcome to your learning hub! Explore, study, and track your progress! 📚✨")

    # Logout button
    if st.button("Logout"):
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.session_state["current_page"] = "Login"
        st.session_state["user_id"] = None
        st.session_state["access_token"] = None
        st.success("Logged out successfully! 😺")
        st.rerun()

    # Progress Overview
    st.subheader("Your Progress Snapshot")
    progress = get_progress(st.session_state.user_id)
    quiz_history = get_quiz_history(st.session_state.user_id)
    if progress or quiz_history:
        total_quizzes = len([p for p in progress if p[1] is not None])  # Count quiz entries
        total_flashcards = sum([p[3] or 0 for p in progress])  # Sum flashcards reviewed
        total_questions = len(quiz_history)
        correct_questions = len([q for q in quiz_history if q[6]])  # is_correct is True
        st.write(f"🎓 **Quizzes Taken**: {total_quizzes}")
        st.write(f"📚 **Flashcards Reviewed**: {total_flashcards}")
        st.write(f"❓ **Questions Answered**: {correct_questions}/{total_questions} ({correct_questions/total_questions*100:.1f}% correct)")
    else:
        st.write("No progress yet! Start a quiz or review flashcards to see your stats! 🚀")

    # Navigation
    st.sidebar.header("Navigation")
    page = st.sidebar.radio("Go to:", [
        "Study Assistant (Q&A)",
        "Study Plan Generator",
        "Quiz Generator",
        "Flashcards Creator",
        "Progress Tracker",
        "Study Suggestions"
    ])

    if page == "Study Assistant (Q&A)":
        run_study_assistant()
    elif page == "Study Plan Generator":
        run_study_plan_generator()
    elif page == "Quiz Generator":
        run_quiz_generator()
    elif page == "Flashcards Creator":
        run_flashcards_creator()
    elif page == "Progress Tracker":
        run_progress_tracker()
    elif page == "Study Suggestions":
        run_study_suggestion()