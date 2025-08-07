-- Initialize SQLite database

-- Create the users table
CREATE TABLE IF NOT EXISTS users (
    user_id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT NOT NULL UNIQUE,
    password TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create the study_plans table
CREATE TABLE IF NOT EXISTS study_plans (
    plan_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    subject TEXT NOT NULL,
    study_date DATE NOT NULL,
    tasks TEXT NOT NULL,
    resources TEXT,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);

-- Create the progress table
CREATE TABLE IF NOT EXISTS progress (
    progress_id INTEGER PRIMARY KEY AUTOINCREMENT,
    user_id INTEGER,
    subject TEXT NOT NULL,
    quiz_score REAL,
    time_spent INTEGER,
    completion_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (user_id) REFERENCES users(user_id)
);
