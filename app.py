import google.generativeai as genai
import os
import streamlit as st
from pydub import AudioSegment

# Configure Google Gemini API
GOOGLE_GEMINI_API_KEY = "YOUR_GEMINI_API_KEY"
genai.configure(api_key=GOOGLE_GEMINI_API_KEY)

# Set custom file size limit (Change as needed)
MAX_FILE_SIZE_MB = 2048  # Change this value to set your limit
MAX_FILE_SIZE_BYTES = MAX_FILE_SIZE_MB * 1024 * 1024

st.title("🎬 AI-Powered Subtitle Search Engine")
st.write(f"📂 **Maximum Upload Size: {MAX_FILE_SIZE_MB}MB**")

uploaded_file = st.file_uploader("Upload an audio file", type=["mp3", "wav", "m4a", "db"])

if uploaded_file is not None:
    # Check file size before saving
    file_size = len(uploaded_file.getbuffer())
    
    if file_size > MAX_FILE_SIZE_BYTES:
        st.error(f"❌ File is too large! Maximum allowed size is {MAX_FILE_SIZE_MB}MB.")
        st.stop()

    st.audio(uploaded_file, format="audio/mp3")

    # Save file to disk
    audio_path = f"temp_{uploaded_file.name}"
    with open(audio_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    st.success(f"✅ File saved as {audio_path} ({file_size / (1024 * 1024):.2f}MB)")

    # Convert audio to WAV format
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
