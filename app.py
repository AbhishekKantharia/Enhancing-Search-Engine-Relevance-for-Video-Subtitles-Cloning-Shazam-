import streamlit as st
import google.generativeai as genai
import chromadb
import whisper
import torch
import os
import json
import numpy as np
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity

# --- Configurations ---
GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Use DuckDB instead of SQLite (fix for Streamlit Cloud)
chroma_client = chromadb.PersistentClient(path="./chroma_db", settings={"chroma_db_impl": "duckdb"})
collection = chroma_client.get_or_create_collection(name="subtitle_embeddings")

# Load Whisper model for speech-to-text conversion
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# --- Helper Functions ---
def preprocess_text(text):
    """Basic cleaning for subtitle text."""
    return text.replace("\n", " ").replace("[", "").replace("]", "").strip()

def chunk_text(text, chunk_size=500, overlap=100):
    """Splits text into overlapping chunks."""
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size - overlap):
        chunks.append(" ".join(words[i : i + chunk_size]))
    return chunks

def get_embedding(text):
    """Uses Google Gemini API to generate text embeddings."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return response["embedding"]

def store_embeddings(subtitle_data):
    """Stores embeddings in ChromaDB."""
    for idx, doc in enumerate(subtitle_data):
        text = preprocess_text(doc["text"])
        chunks = chunk_text(text)
        for chunk in chunks:
            embedding = get_embedding(chunk)
            collection.add(
                ids=[f"doc_{idx}_{chunk[:10]}"],  # Unique ID
                embeddings=[embedding],
                metadatas=[{"text": chunk}]
            )

def transcribe_audio(audio_path):
    """Transcribes audio into text using Whisper."""
    audio = whisper.load_audio(audio_path)
    result = whisper_model.transcribe(audio)
    return result["text"]

def search_subtitles(query_text):
    """Searches subtitles using embeddings and cosine similarity."""
    query_embedding = get_embedding(query_text)
    results = collection.query(
        query_embeddings=[query_embedding], 
        n_results=5  # Fetch top 5 results
    )
    
    # Sort results by cosine similarity
    sorted_results = sorted(
        zip(results["ids"], results["distances"], results["metadatas"]),
        key=lambda x: x[1], reverse=True
    )

    return [res[2]["text"] for res in sorted_results]

# --- Streamlit UI ---
st.title("Video Subtitle Search Engine")

uploaded_file = st.file_uploader("Upload a 2-minute audio clip", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    # Convert audio to WAV format if needed
    audio = AudioSegment.from_file(uploaded_file)
    audio_path = "temp_audio.wav"
    audio.export(audio_path, format="wav")

    # Transcribe and search
    with st.spinner("Transcribing audio..."):
        query_text = transcribe_audio(audio_path)
    
    st.write("Transcribed Text:", query_text)

    with st.spinner("Searching subtitles..."):
        results = search_subtitles(query_text)

    st.subheader("Top Matching Subtitles:")
    for idx, result in enumerate(results):
        st.write(f"**{idx + 1}.** {result}")
