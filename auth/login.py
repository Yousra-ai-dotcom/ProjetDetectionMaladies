import hashlib

def authenticate(conn, username, password):
    c = conn.cursor()
    c.execute("SELECT password FROM users WHERE username=?", (username,))
    result = c.fetchone()
    if result:
        hashed = hashlib.sha256(password.encode()).hexdigest()
        return hashed == result[0]
    return False