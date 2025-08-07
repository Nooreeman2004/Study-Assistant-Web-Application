import streamlit as st
import sqlite3
import pandas as pd
import plotly.express as px
import json
from db_utils import get_progress, get_quiz_history, get_chat_history, get_study_plans
from middleware import protect_page

def run_progress_tracker():
    protect_page()  # Validate JWT before rendering page
    st.header("Progress Tracker 📊")
    st.write("Track your learning journey with cute graphs and motivational feedback! 🌟😺")

    username = st.session_state.user_id
    if not username:
        st.error("Please log in to view your progress.")
        return

    conn = sqlite3.connect("study_assistant.db")
    c = conn.cursor()

    # Fetch progress data
    c.execute("""
        SELECT topic, quiz_score, quiz_total, flashcards_reviewed, timestamp
        FROM progress
        WHERE username = ? AND (quiz_score IS NOT NULL OR flashcards_reviewed IS NOT NULL)
        ORDER BY timestamp DESC
    """, (username,))
    progress_data = c.fetchall()

    # Fetch quiz history
    c.execute("""
        SELECT topic, question, question_type, options, correct_answer, user_answer, is_correct, timestamp
        FROM quiz_history
        WHERE username = ?
        ORDER BY timestamp DESC
    """, (username,))
    quiz_history = c.fetchall()
    conn.close()

    if not progress_data and not quiz_history:
        st.write("No progress data yet. Start studying to see your progress! 🚀")
        return

    # Quiz Score Trends
    if progress_data:
        df = pd.DataFrame(progress_data, columns=["Topic", "Quiz Score", "Quiz Total", "Flashcards Reviewed", "Timestamp"])
        df["Score Percentage"] = (df["Quiz Score"] / df["Quiz Total"] * 100).fillna(0)
        df["Timestamp"] = pd.to_datetime(df["Timestamp"])

        st.subheader("Quiz Score Trends")
        fig = px.line(df, x="Timestamp", y="Score Percentage", color="Topic", title="Quiz Performance Over Time")
        fig.update_layout(
            plot_bgcolor="#000000",
            paper_bgcolor="#000000",
            font_color="#FFFFFF",
            title_font_color="#FF69B4",
            xaxis_gridcolor="#555555",
            yaxis_gridcolor="#555555"
        )
        st.plotly_chart(fig)

        # Flashcard Reviews
        df_flashcards = df[df["Flashcards Reviewed"].notnull()]
        if not df_flashcards.empty:
            st.subheader("Flashcards Reviewed")
            fig = px.bar(df_flashcards, x="Timestamp", y="Flashcards Reviewed", color="Topic", title="Flashcards Reviewed Over Time")
            fig.update_layout(
                plot_bgcolor="#000000",
                paper_bgcolor="#000000",
                font_color="#FFFFFF",
                title_font_color="#FF69B4",
                xaxis_gridcolor="#555555",
                yaxis_gridcolor="#555555"
            )
            st.plotly_chart(fig)

        # Motivational Feedback
        latest_score = df["Score Percentage"].iloc[0] if not df.empty else 0
        if latest_score >= 80:
            st.balloons()
            st.write("Wow, you're a study superstar! Keep shining! 🌟")
        elif latest_score >= 50:
            st.write("Great effort! You're making progress, keep it up! 😺")
        else:
            st.write("Don't worry, every step counts! Let's tackle those topics together! 🚀")

    # Detailed Quiz History
    if quiz_history:
        st.subheader("Quiz History 📝")
        df_quiz = pd.DataFrame(quiz_history, columns=["Topic", "Question", "Type", "Options", "Correct Answer", "User Answer", "Is Correct", "Timestamp"])
        df_quiz["Timestamp"] = pd.to_datetime(df_quiz["Timestamp"])
        df_quiz["Is Correct"] = df_quiz["Is Correct"].map({1: "✅ Correct", 0: "❌ Incorrect"})

        for topic in df_quiz["Topic"].unique():
            st.write(f"**Topic: {topic}**")
            topic_quizzes = df_quiz[df_quiz["Topic"] == topic]
            for _, row in topic_quizzes.iterrows():
                with st.expander(f"Question (Taken on {row['Timestamp'].strftime('%Y-%m-%d %H:%M:%S')}): {row['Question']}"):
                    st.write(f"**Type**: {row['Type']}")
                    if row["Type"] == "multiple_choice":
                        options = json.loads(row["Options"])
                        st.write("**Options**: " + ", ".join(options))
                    st.write(f"**Your Answer**: {row['User Answer']}")
                    st.write(f"**Correct Answer**: {row['Correct Answer']}")
                    st.write(f"**Status**: {row['Is Correct']}")
            st.write("---")

    # Summary Statistics
    if quiz_history:
        total_questions = len(df_quiz)
        correct_questions = len(df_quiz[df_quiz["Is Correct"] == "✅ Correct"])
        st.write(f"**Overall Stats**: {correct_questions}/{total_questions} questions correct ({correct_questions/total_questions*100:.1f}%)")