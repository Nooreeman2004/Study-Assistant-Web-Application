import sqlite3
import os

# Forcefully close any connections to the database
try:
    conn = sqlite3.connect("study_assistant.db")
    conn.close()
except Exception as e:
    print(f"Error closing database: {e}")

# Remove the database file
if os.path.exists("study_assistant.db"):
    os.remove("study_assistant.db")
    print("Database removed successfully!")
else:
    print("Database file not found.")

