
import sqlite3
from werkzeug.security import generate_password_hash

# Connect to SQLite database (will create if it doesn't exist)
conn = sqlite3.connect('admin.db')
cursor = conn.cursor()

# Create admin table
cursor.execute('''
CREATE TABLE IF NOT EXISTS admin (
    username TEXT PRIMARY KEY,
    password_hash TEXT NOT NULL
)
''')

# Default admin credentials
username = 'admin'
password = 'admin123'
hashed_pw = generate_password_hash(password)

# Insert only if not exists
cursor.execute('SELECT * FROM admin WHERE username = ?', (username,))
if not cursor.fetchone():
    cursor.execute('INSERT INTO admin (username, password_hash) VALUES (?, ?)', (username, hashed_pw))
    print("✅ Default admin user created.")
else:
    print("ℹ️ Admin user already exists.")

conn.commit()
conn.close()
