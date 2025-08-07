import jwt
import datetime
import os
from dotenv import load_dotenv

load_dotenv()


def generate_access_token(username):
    """Generate a JWT access token valid for 15 minutes."""
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(minutes=15),
        "iat": datetime.datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def generate_refresh_token(username):
    """Generate a JWT refresh token valid for 7 days."""
    payload = {
        "username": username,
        "exp": datetime.datetime.utcnow() + datetime.timedelta(days=7),
        "iat": datetime.datetime.utcnow()
    }
    return jwt.encode(payload, SECRET_KEY, algorithm=ALGORITHM)

def validate_token(token):
    """Validate a JWT token and return the payload or None if invalid."""
    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        return None
    except jwt.InvalidTokenError:
        return None

def get_token_from_session():
    """Retrieve access token from session state."""
    return st.session_state.get("access_token")