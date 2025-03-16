import google.generativeai as genai
import os
import streamlit as st
from pydub import AudioSegment

# Configure Google Gemini API
GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

st.title("🎬 AI-Powered Subtitle Search Engine")

# Increase file uploader size
st.write("🔹 Maximum Upload Size: 1GB")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a"])

if uploaded_file is not None:
    st.audio(uploaded_file, format="audio/mp3")

    # Save file to disk
    audio_path = f"temp_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File saved as {audio_path}")

    # Convert audio to WAV format (for Whisper model compatibility)
    audio = AudioSegment.from_file(audio_path)
    wav_path = audio_path.replace(".mp3", ".wav")
    audio.export(wav_path, format="wav")

    # Transcribe the audio
    with st.spinner("Transcribing audio using Google Gemini..."):
        query_text = genai.transcribe(model="models/whisper-large", audio=open(wav_path, "rb").read())

    # Display transcription
    if "transcription" in query_text:
        st.write("**Transcribed Text:**", query_text["transcription"])
    else:
        st.error("❌ Transcription failed.")
