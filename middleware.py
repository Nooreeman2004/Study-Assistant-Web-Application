import streamlit as st
from auth_utils import validate_token, get_token_from_session

def protect_page():
    """Check if the user is authenticated with a valid JWT; redirect to login if not."""
    access_token = get_token_from_session()
    if not access_token:
        st.error("Please log in to access this page.")
        st.session_state.current_page = "Login"
        st.rerun()
    payload = validate_token(access_token)
    if not payload or payload["username"] != st.session_state.get("user_id"):
        st.error("Invalid or expired session. Please log in again.")
        st.session_state.current_page = "Login"
        st.rerun()