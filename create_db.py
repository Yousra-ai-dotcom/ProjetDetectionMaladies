import sqlite3

conn = sqlite3.connect('users.db')
c = conn.cursor()

# Table utilisateurs
c.execute('''
CREATE TABLE IF NOT EXISTS users (
    username TEXT PRIMARY KEY,
    password TEXT NOT NULL
)
''')

# Table historique analyses
c.execute('''
CREATE TABLE IF NOT EXISTS history (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    username TEXT,
    image_path TEXT,
    health_status TEXT,
    disease_detected TEXT,
    confidence REAL,
    analysis_date TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(username) REFERENCES users(username)
)
''')

conn.commit()
conn.close()
print("Base users.db créée.")