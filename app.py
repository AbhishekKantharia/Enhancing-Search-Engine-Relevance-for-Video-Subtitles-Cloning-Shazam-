import streamlit as st
import sqlite3
import faiss
import numpy as np
import google.generativeai as genai
import pandas as pd
import pickle
import os
import zlib

# --- Configurations ---
DB_FILE = "eng_subtitles_database.db"
GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
TABLE_NAME = "zipfiles"  # Confirmed table name
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# FAISS Index Configuration
VECTOR_DIMENSION = 768  # Gemini embedding size
INDEX_FILE = "faiss_index.pkl"
index = faiss.IndexFlatL2(VECTOR_DIMENSION)
metadata = {}

# --- Step 1: Load Subtitle Data ---
def load_subtitles(limit=5000):
    """Loads compressed subtitles from the database."""
    conn = sqlite3.connect(DB_FILE)
    query = f"SELECT num, name, content FROM {TABLE_NAME} LIMIT {limit}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    subtitles = []
    for _, row in df.iterrows():
        subtitle_id = row["num"]
        file_name = row["name"]
        binary_content = row["content"]

        try:
            # Decompress and decode subtitle text
            decoded_text = zlib.decompress(binary_content).decode("latin-1")
            subtitles.append((subtitle_id, file_name, decoded_text))
        except Exception as e:
            print(f"Error decoding subtitle {file_name}: {e}")
    
    return subtitles

# --- Step 2: Preprocess Subtitle Data ---
def clean_subtitle(text):
    """Removes timestamps and unnecessary lines."""
    lines = text.split("\n")
    cleaned_lines = [line.strip() for line in lines if "-->" not in line]
    return " ".join(cleaned_lines)

# --- Step 3: Generate Embeddings ---
def get_embedding(text):
    """Uses Google Gemini API to generate text embeddings."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"]).reshape(1, -1)

# --- Step 4: Store Embeddings in FAISS ---
def store_embeddings(subtitle_data):
    """Stores embeddings in FAISS for fast retrieval."""
    global index, metadata

    for subtitle_id, file_name, content in subtitle_data:
        cleaned_text = clean_subtitle(content)
        embedding = get_embedding(cleaned_text)

        # Store in FAISS
        index.add(embedding)
        metadata[len(metadata)] = {
            "subtitle_id": subtitle_id,
            "file_name": file_name,
            "text": cleaned_text,
        }

    # Save FAISS index and metadata
    with open(INDEX_FILE, "wb") as f:
        pickle.dump((index, metadata), f)

    print("✅ Stored subtitle embeddings in FAISS!")

# --- Load and Process Data (Run Once) ---
if not os.path.exists(INDEX_FILE):
    subtitles = load_subtitles(limit=5000)
    store_embeddings(subtitles)
else:
    with open(INDEX_FILE, "rb") as f:
        index, metadata = pickle.load(f)

# --- Step 5: Search Function ---
def search_subtitles(query, top_k=5):
    """Searches for the most relevant subtitles using FAISS."""
    query_embedding = get_embedding(query)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(len(indices[0])):
        meta = metadata.get(indices[0][i], {})
        if meta:
            results.append({
                "score": distances[0][i],
                "subtitle_id": meta["subtitle_id"],
                "file_name": meta["file_name"],
                "text": meta["text"][:500],  # Show only first 500 characters
                "link": f"https://www.opensubtitles.org/en/subtitles/{meta['subtitle_id']}"
            })

    return results

# --- Step 6: Streamlit UI ---
st.title("🎬 AI-Powered Subtitle Search Engine")
st.markdown("**Search for subtitles using natural language queries!**")

query = st.text_input("Enter a movie scene description:")

if st.button("Search"):
    with st.spinner("Searching..."):
        results = search_subtitles(query)

    st.subheader("🔍 Top Matching Subtitles:")
    if results:
        for res in results:
            st.markdown(f"**🎬 {res['file_name']}**")
            st.markdown(f"🔗 [View on OpenSubtitles]({res['link']})")
            st.write(f"📜 {res['text']}...")
            st.write(f"🔥 Score: {res['score']}")
    else:
        st.warning("No matching subtitles found!")
