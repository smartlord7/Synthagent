import sqlite3
import os
from datetime import datetime
import json

_HERE = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_HERE, "llm_logs.db")

def init_db():
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS logs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            algorithm TEXT,
            prompt TEXT,
            response TEXT,
            status TEXT
        )
    ''')
    conn.commit()
    conn.close()

def log_interaction(algorithm, prompt, response, status="success"):
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute(
        "INSERT INTO logs (timestamp, algorithm, prompt, response, status) VALUES (?, ?, ?, ?, ?)",
        (datetime.now().isoformat(), algorithm, prompt, response, status)
    )
    conn.commit()
    conn.close()

init_db()