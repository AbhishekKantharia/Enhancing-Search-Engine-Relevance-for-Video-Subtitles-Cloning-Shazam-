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
    
    # Extract the transcribed text
    if "transcription" in response:
        return response["transcription"]
    else:
        return "Transcription failed."

st.title("🎬 AI-Powered Subtitle Search Engine")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "db"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3/db")

    # Convert audio to WAV format
    audio = AudioSegment.from_file(uploaded_file)
    audio_path = "temp_audio.mp3"
    audio.export(audio_path, format="mp3")

    with st.spinner("Transcribing audio using Google Gemini..."):
        query_text = transcribe_audio_google(audio_path)
    
    st.write("**Transcribed Text:**", query_text)
