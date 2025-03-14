import streamlit as st
import faiss
import numpy as np
import google.generativeai as genai
import whisper
import torch
import os
from pydub import AudioSegment
from sklearn.metrics.pairwise import cosine_similarity
import pickle

# --- Configurations ---
GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Load Whisper model
device = "cuda" if torch.cuda.is_available() else "cpu"
whisper_model = whisper.load_model("base", device=device)

# Initialize FAISS index
INDEX_FILE = "faiss_index.pkl"
VECTOR_DIMENSION = 768  # Gemini embedding size
index = faiss.IndexFlatL2(VECTOR_DIMENSION)

# Load stored embeddings if they exist
if os.path.exists(INDEX_FILE):
    with open(INDEX_FILE, "rb") as f:
        index = pickle.load(f)

# --- Helper Functions ---
def preprocess_text(text):
    """Clean text by removing unnecessary characters."""
    return text.replace("\n", " ").strip()

def get_embedding(text):
    """Generate embeddings using Google Gemini API."""
    response = genai.embed_content(
        model="models/embedding-001",
        content=text,
        task_type="retrieval_document"
    )
    return np.array(response["embedding"]).reshape(1, -1)

def store_embeddings(subtitle_data):
    """Stores embeddings in FAISS."""
    global index
    all_vectors = []
    all_texts = []

    for doc in subtitle_data:
        text = preprocess_text(doc["text"])
        embedding = get_embedding(text)
        all_vectors.append(embedding)
        all_texts.append(text)

    all_vectors = np.vstack(all_vectors)
    index.add(all_vectors)

    # Save FAISS index
    with open(INDEX_FILE, "wb") as f:
        pickle.dump(index, f)

    return all_texts

def transcribe_audio(audio_path):
    """Transcribes audio into text using Whisper."""
    audio = whisper.load_audio(audio_path)
    result = whisper_model.transcribe(audio)
    return result["text"]

def search_subtitles(query_text, top_k=5):
    """Searches for the most relevant subtitles using FAISS."""
    query_embedding = get_embedding(query_text)
    distances, indices = index.search(query_embedding, top_k)
    
    results = []
    for i in range(len(indices[0])):
        results.append(f"Match {i+1}: Score={distances[0][i]}")

    return results

# --- Streamlit UI ---
import google.generativeai as genai
import os
import streamlit as st
from pydub import AudioSegment

# Configure Google Gemini API
GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

def transcribe_audio_google(audio_path):
    """Transcribes audio using Google Gemini's Whisper model."""
    with open(audio_path, "rb") as audio_file:
        response = genai.transcribe(model="models/whisper-large", audio=audio_file.read())
    return response["transcription"] if "transcription" in response else "Transcription failed."

st.title("🎬 AI-Powered Subtitle Search Engine")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    # Convert audio to WAV format
    audio = AudioSegment.from_file(uploaded_file)
    audio_path = "temp_audio.wav"
    audio.export(audio_path, format="wav")

    with st.spinner("Transcribing audio using Google Gemini..."):
        query_text = transcribe_audio_google(audio_path)
    
    st.write("**Transcribed Text:**", query_text)

    st.subheader("Top Matching Subtitles:")

    for result in results:
        st.write(result)
