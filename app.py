import os
import sqlite3
import requests
import streamlit as st

# Google Drive File ID
GDRIVE_FILE_ID = "1bKx176TVlxQbMEFuDyzSBmceLapQYHT8"
DB_FILE = "eng_subtitles_database.db"

# Step 1: Check if the database exists, if not, download it
def download_database():
    """Downloads the database from Google Drive if it does not exist."""
    if not os.path.exists(DB_FILE):
        st.write("📥 Downloading database file from Google Drive...")
        url = f"https://drive.google.com/uc?export=download&id={GDRIVE_FILE_ID}"
        
        response = requests.get(url, stream=True)
        if response.status_code == 200:
            with open(DB_FILE, "wb") as db_file:
                for chunk in response.iter_content(chunk_size=1024):
                    db_file.write(chunk)
            st.success("✅ Database downloaded successfully!")
        else:
            st.error("❌ Failed to download the database. Please check the Google Drive link.")
            st.stop()

# Step 2: Run the database download check
download_database()

# Step 3: Verify if the database file size is valid
def check_database_file():
    """Checks if the database file is valid (not empty)."""
    if not os.path.exists(DB_FILE):
        st.error("❌ Database file does not exist!")
        st.stop()

    file_size = os.path.getsize(DB_FILE)
    st.write(f"📂 Database file size: {file_size} bytes")
    
    if file_size < 1000:  # Too small for a real database
        st.error("❌ ERROR: The database file is too small! The download may have failed.")
        st.stop()

check_database_file()

# Step 4: Check if the database contains tables
def check_database():
    """Checks if the database contains any tables."""
    conn = sqlite3.connect(DB_FILE)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    conn.close()

    if not tables:
        st.error("❌ ERROR: The database is empty! No tables found.")
        st.stop()
    
    st.write(f"✅ Found Tables: {tables}")
    return tables[0][0]  # Use the first available table

TABLE_NAME = check_database()
