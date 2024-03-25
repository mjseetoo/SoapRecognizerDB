import psycopg2
from psycopg2 import sql

# Connect to PostgreSQL database
conn = psycopg2.connect(
    dbname="SoapRecognizerDB",
    user="postgres",
    password="miker",
    host="127.0.0.1",
    port="5432"
)
cursor = conn.cursor()

# Create a table to store image metadata if it doesn't exist
cursor.execute("""
    CREATE TABLE IF NOT EXISTS images (
        id SERIAL PRIMARY KEY,
        filename VARCHAR NOT NULL,
        user_id INTEGER NOT NULL,
        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
    )
""")
conn.commit()

# Function to upload image metadata to the database
def upload_image(filename, user_id):
    cursor.execute("""
        INSERT INTO images (filename, user_id) VALUES (%s, %s)
    """, (filename, user_id))
    conn.commit()

# Example usage: Upload image metadata
#upload_image('example.jpg', 1)

# Close the database connection when done
cursor.close()
conn.close()