import streamlit as st
import sqlite3
import pandas as pd
from dotenv import load_dotenv
import os
import json
import re
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate
import requests
from bs4 import BeautifulSoup
from middleware import protect_page

def run_study_suggestion():
    protect_page()  # Validate JWT before rendering page
    # Load environment variables
    load_dotenv()
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        st.error("GROQ_API_KEY not found in .env file. Please add it and restart the app.")
        st.stop()

    # Streamlit app title and description
    st.title("Study Suggestions 📚")
    st.write("Get personalized recommendations for your next study topic based on your progress! 🌟")

    username = st.session_state.get("user_id")
    if not username:
        st.error("Please log in to view study suggestions! 😺")
        return

    # Fetch progress data from database
    def get_progress_data():
        conn = sqlite3.connect("study_assistant.db")
        query = """
            SELECT topic, quiz_score, quiz_total, flashcards_reviewed, timestamp
            FROM progress
            WHERE username = ?
            ORDER BY timestamp
        """
        df = pd.read_sql_query(query, conn, params=(username,))
        conn.close()
        return df

    df = get_progress_data()

    if df.empty:
        st.warning("No progress data yet! Start by completing quizzes or reviewing flashcards to get suggestions! 🚀")
        return

    # Analyze progress
    def analyze_progress(df):
        # Calculate quiz performance
        if 'quiz_score' in df and 'quiz_total' in df:
            df['quiz_percentage'] = (df['quiz_score'] / df['quiz_total']) * 100
            weak_topics = df[df['quiz_percentage'] < 70]['topic'].tolist()
        else:
            weak_topics = []

        # Get all studied topics
        studied_topics = df['topic'].unique().tolist()

        # Count flashcards reviewed per topic
        flashcard_counts = df.groupby('topic')['flashcards_reviewed'].sum().to_dict()

        return weak_topics, studied_topics, flashcard_counts

    weak_topics, studied_topics, flashcard_counts = analyze_progress(df)

    # Generate suggestion using LangChain
    def generate_suggestion(weak_topics, studied_topics, flashcard_counts):
        llm = ChatGroq(groq_api_key=groq_api_key, model_name="llama3-8b-8192", temperature=0.3)
        prompt_template = """
        You are an expert study advisor. Based on the user's study progress, recommend the next topic or concept they should study. Follow these guidelines:

        - Studied topics: {studied_topics}
        - Weak topics (low quiz scores): {weak_topics}
        - Flashcards reviewed per topic: {flashcard_counts}
        - If weak topics exist, prioritize one of them for review.
        - If no weak topics, suggest a new topic that logically follows the studied topics or complements them.
        - Provide a brief explanation (1-2 sentences) for why this topic is recommended.
        - Suggest 1-2 specific concepts within the topic to focus on.

        Output only a valid JSON object with the following structure, with no additional text or comments:
        {{
            "topic": "Recommended topic",
            "explanation": "Why this topic is recommended",
            "concepts": ["Concept 1", "Concept 2"]
        }}
        """
        prompt = PromptTemplate(
            input_variables=["studied_topics", "weak_topics", "flashcard_counts"],
            template=prompt_template
        )

        response = llm.invoke(prompt.format(
            studied_topics=", ".join(studied_topics) if studied_topics else "None",
            weak_topics=", ".join(weak_topics) if weak_topics else "None",
            flashcard_counts=str(flashcard_counts)
        ))

        try:
            # Extract JSON using regex to handle extraneous text
            json_match = re.search(r'\{.*\}', response.content, re.DOTALL)
            if not json_match:
                st.error("Failed to extract JSON from response.")
                return None

            json_str = json_match.group(0)
            suggestion = json.loads(json_str)
            return suggestion
        except json.JSONDecodeError as e:
            st.error(f"Failed to parse suggestion: {e}")
            st.write("Debug: Invalid JSON extracted:", json_str)
            return None

    # Web scraping for resources (optional)
    def fetch_web_resources(topic):
        try:
            # Example: Search Wikipedia for the topic
            url = f"https://en.wikipedia.org/wiki/{topic.replace(' ', '_')}"
            response = requests.get(url)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                # Extract first paragraph for a brief description
                intro = soup.find('p', class_=False).text.strip()[:200] + "..."
                return [{"title": f"Wikipedia: {topic}", "url": url, "description": intro}]
            return []
        except Exception as e:
            st.warning(f"Could not fetch web resources: {e}")
            return []

    # Generate and display suggestion
    with st.spinner("Generating study suggestion..."):
        suggestion = generate_suggestion(weak_topics, studied_topics, flashcard_counts)
        if suggestion:
            st.subheader("Your Next Study Topic")
            st.write(f"**Topic**: {suggestion['topic']}")
            st.write(f"**Why?**: {suggestion['explanation']}")
            st.write(f"**Focus Concepts**: {', '.join(suggestion['concepts'])}")

            # Fetch web resources
            resources = fetch_web_resources(suggestion['topic'])
            if resources:
                st.subheader("Recommended Resources")
                for res in resources:
                    st.write(f"[{res['title']}]({res['url']})")
                    st.write(res['description'])

    # Motivational message
    st.markdown("---")
    st.write("You're on the right track! Dive into this topic and keep shining! 😺💖")