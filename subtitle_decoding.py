import os
import sqlite3
import zipfile
import io
import pandas as pd
import re

# Constants
DB_PATH = "eng_subtitles_database.db"
DATASET_DIR = "dataset"

# Ensure dataset directory exists
os.makedirs(DATASET_DIR, exist_ok=True)

def connect_database(db_path):
    """Connects to the SQLite database."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(f"❌ Database file not found: {db_path}")
    return sqlite3.connect(db_path)

def fetch_subtitle_data(con):
    """Fetches subtitle data from the database."""
    query = "SELECT name, content FROM zipfiles"
    return pd.read_sql_query(query, con)

def extract_subtitle(zip_binary):
    """Extracts and decodes subtitle content from a ZIP binary."""
    try:
        with io.BytesIO(zip_binary) as file:
            with zipfile.ZipFile(file, "r") as zip_file:
                subtitle_filename = zip_file.namelist()[0]  # Extract the first file
                subtitle_content = zip_file.read(subtitle_filename)
                return subtitle_content.decode("latin-1")  # Adjust encoding if needed
    except Exception as e:
        print(f"❌ Error extracting subtitle: {e}")
        return None

def clean_subtitle_text(subtitle_text):
    """Cleans subtitle text by removing timestamps and extra spaces."""
    if not subtitle_text:
        return ""
    
    # Remove timestamps (e.g., "00:00:00,000 --> 00:00:05,000")
    subtitle_text = re.sub(r"\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}", "", subtitle_text)
    
    # Remove extra spaces and empty lines
    return "\n".join(line.strip() for line in subtitle_text.split("\n") if line.strip())

def save_subtitle_file(movie_name, subtitle_text):
    """Saves cleaned subtitle text as an .srt file."""
    if not subtitle_text:
        print(f"⚠️ No subtitle content for {movie_name}, skipping...")
        return
    
    file_path = os.path.join(DATASET_DIR, f"{movie_name}_subtitle_cleaned.srt")
    with open(file_path, "w", encoding="utf-8") as file:
        file.write(subtitle_text)
    
    print(f"✅ Saved: {file_path}")

def process_subtitles(start=0, end=10):
    """Processes and saves subtitles from the database within a given range."""
    con = connect_database(DB_PATH)
    df = fetch_subtitle_data(con)
    
    for i, (movie_name, zip_binary) in enumerate(zip(df["name"][start:end], df["content"][start:end]), start=start):
        subtitle_text = extract_subtitle(zip_binary)
        cleaned_text = clean_subtitle_text(subtitle_text)
        save_subtitle_file(movie_name, cleaned_text)

    con.close()

# Run subtitle processing for the first 10 entries
process_subtitles(0, 10)
