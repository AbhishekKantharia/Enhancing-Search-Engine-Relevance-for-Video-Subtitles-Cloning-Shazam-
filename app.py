import os
import sys
import streamlit as st
import torch
import torchaudio
import whisper
import pysqlite3  # ✅ Fix for ChromaDB sqlite3 issue
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.document_loaders import DirectoryLoader, TextLoader
from langchain_community.vectorstores import Chroma
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA

# ✅ Override sqlite3 module to avoid ChromaDB issues
sys.modules["sqlite3"] = pysqlite3

# ✅ Constants
CHROMA_DB_DIR = "./chroma_db"
DATASET_DIR = "dataset"
WHISPER_MODEL_NAME = "base"
HUGGINGFACE_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
GOOGLE_API_KEY = "YOUR_GEMINI_API_KEY"  # 🔹 Replace with a secure way to store API keys

# ✅ Ensure required directories exist
os.makedirs(CHROMA_DB_DIR, exist_ok=True)
os.makedirs(DATASET_DIR, exist_ok=True)


# ✅ Lazy Initialization Functions
@st.cache_resource
def load_whisper_model():
    """Loads the Whisper model for speech-to-text conversion."""
    return whisper.load_model(WHISPER_MODEL_NAME)


@st.cache_resource
def initialize_llm():
    """Initializes the Gemini LLM using an API key."""
    try:
        return ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)
    except Exception as e:
        st.error(f"❌ Error initializing LLM: {e}")
        return None


@st.cache_resource
def load_retriever():
    """Loads subtitle files and stores them in ChromaDB with embeddings."""
    try:
        loader = DirectoryLoader(DATASET_DIR, glob="*_cleaned.srt", loader_cls=TextLoader)
        documents = loader.load()
        
        # 🔹 Improved Chunking for better retrieval context
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
        texts = text_splitter.split_documents(documents)
        
        embeddings = HuggingFaceEmbeddings(model_name=HUGGINGFACE_MODEL_NAME)

        # Initialize ChromaDB with persistence
        db = Chroma.from_documents(texts, embeddings, persist_directory=CHROMA_DB_DIR)
        db.persist()
        
        return db
    except Exception as e:
        st.error(f"❌ Error loading retriever: {e}")
        return None


# ✅ Audio Processing Function
def convert_audio_to_text(audio_file, model):
    """Converts uploaded audio to text using Whisper."""
    try:
        waveform, sample_rate = torchaudio.load(audio_file)
        audio = waveform.squeeze().numpy()
        audio = whisper.pad_or_trim(audio)
        mel = whisper.log_mel_spectrogram(audio).to(model.device)
        options = whisper.DecodingOptions(fp16=False)
        result = whisper.decode(model, mel, options)
        return result.text
    except Exception as e:
        st.error(f"❌ Error in audio transcription: {e}")
        return ""


# ✅ Query LLM for additional context
def query_llm(llm, query):
    """Queries Gemini AI for context on retrieved subtitles."""
    if llm is None:
        return "⚠️ LLM Not Available"
    
    try:
        return llm.invoke(query)
    except Exception as e:
        st.warning(f"⚠️ LLM Query Failed: {e}")
        return "No additional context available."


# ✅ Streamlit UI
def main():
    st.title("🎬 AI-Powered Subtitle Search")
    st.write("Upload a **TV/Movie audio clip**, and AI will find matching subtitles.")

    whisper_model = load_whisper_model()
    llm = initialize_llm()
    retriever = load_retriever()

    uploaded_file = st.file_uploader("📤 Upload Audio File", type=["mp3", "wav", "m4a"])

    if uploaded_file is not None:
        with st.spinner("🎙️ Transcribing audio..."):
            transcription = convert_audio_to_text(uploaded_file, whisper_model)
        
        if transcription:
            st.text_area("📝 Transcription:", transcription, height=150)

            with st.spinner("🔍 Searching subtitles..."):
                docs = retriever.similarity_search(transcription, k=3) if retriever else []

            # ✅ Display retrieved subtitles and context
            st.write("### 🎞️ Retrieved Subtitles and Context")
            if docs:
                for doc in docs:
                    movie_name = doc.metadata.get("source", "Unknown Movie")
                    subtitle_text = doc.page_content

                    llm_query = f"Extract key details from this subtitle: {subtitle_text}"
                    context = query_llm(llm, llm_query)

                    st.subheader(f"🎬 {movie_name}")
                    st.write(f"**📜 Subtitle:** {subtitle_text}")
                    st.write(f"**🤖 Context:** {context}")
            else:
                st.warning("⚠️ No relevant subtitles found.")

if __name__ == "__main__":
    main()
