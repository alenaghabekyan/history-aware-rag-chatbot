import sqlite3
from pathlib import Path
DB_PATH = Path(__file__).resolve().parent.parent/"chat_history.db"


def init_db():
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS messages (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            session_id TEXT NOT NULL,
            role TEXT NOT NULL,
            content TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    cursor.execute("""
    CREATE TABLE IF NOT EXISTS conversation_summaries (
        session_id TEXT PRIMARY KEY,
        summary TEXT NOT NULL
    )
""")
    connection.commit()
    connection.close()

def save_message(session_id, role, content):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()
    cursor.execute("""
        INSERT INTO messages (session_id, role, content)
        VALUES (?, ?, ?)
    """, (session_id, role, content))


    connection.commit()
    connection.close()


def get_history(session_id):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY id ASC
    """, (session_id,))

    rows = cursor.fetchall()

    connection.close()

    chat_history = []

    for role, content in rows:

        chat_history.append({
            "role": role,
            "content": content
        })

    return chat_history


def clear_history(session_id):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM messages
        WHERE session_id = ?
    """, (session_id,))
    cursor.execute("""
    DELETE FROM conversation_summaries
    WHERE session_id = ?
""", (session_id,))

    connection.commit()
    connection.close()



def get_recent_history(session_id, limit=10):

    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT role, content
        FROM messages
        WHERE session_id = ?
        ORDER BY id DESC
        LIMIT ?
    """, (session_id, limit))

    rows = cursor.fetchall()

    connection.close()

    rows.reverse()

    return [
        {
            "role": role,
            "content": content
        }
        for role, content in rows
    ]


def get_summary(session_id):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        SELECT summary
        FROM conversation_summaries
        WHERE session_id = ?
    """, (session_id,))

    row = cursor.fetchone()
    connection.close()

    return row[0] if row else ""


def save_summary(session_id, summary):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO conversation_summaries
        (session_id, summary)
        VALUES (?, ?)
    """, (session_id, summary))

    connection.commit()
    connection.close()


def keep_recent_messages(session_id, limit=6):
    connection = sqlite3.connect(DB_PATH)
    cursor = connection.cursor()

    cursor.execute("""
        DELETE FROM messages
        WHERE session_id = ?
        AND id NOT IN (
            SELECT id
            FROM messages
            WHERE session_id = ?
            ORDER BY id DESC
            LIMIT ?
        )
    """, (session_id, session_id, limit))

    connection.commit()
    connection.close()