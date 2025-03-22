# 🎙️ Movie Subtitle Search via Audio (Whisper + ChromaDB + Gemini AI)

This project transcribes audio files using OpenAI's **Whisper**, then searches for relevant **movie subtitles** stored in a **ChromaDB** vector database. **Google Gemini AI** enhances subtitle extraction and matching.

---

## 🚀 Steps to Run the Project

### **1️⃣ Install Dependencies**
Run the following command to install all required Python libraries:

```sh
pip install torch whisper sentence-transformers chromadb numpy scikit-learn google-generativeai
```

---

### **2️⃣ Install & Verify FFmpeg (Required for Whisper)**
Whisper requires **FFmpeg** to process audio files. Install and check it using:

```sh
sudo apt update
sudo apt install ffmpeg
ffmpeg -version  # Ensure FFmpeg is installed
```

---

### **3️⃣ Set Up ChromaDB for Subtitle Storage**
Initialize **ChromaDB** for efficient search:

```python
import chromadb

chroma_client = chromadb.PersistentClient(path="./subtitle_chromadb")
collection = chroma_client.create_collection(name="subtitles")
```

---

### **4️⃣ Transcribe an Audio File**
Use OpenAI's Whisper model to convert speech to text:

```python
transcript = transcribe_audio_whisper("your_audio_file.mp3")
print(transcript)
```

---

### **5️⃣ Search for Matching Movies Based on Subtitles**

Retrieve the top **5** most similar movies:

```python
find_movie_by_subtitle(transcript, top_k=5)
```

---

### **6️⃣ Enable Google Gemini AI for Improved Matching**

Sign up for **Google Gemini API**, get your API key, and configure it:

```python
import google.generativeai as genai
genai.configure(api_key="YOUR_GEMINI_API_KEY")
```

---

### **7️⃣ Convert Audio Format (If Needed)**

If your audio file isn't working, convert it to a compatible format using FFmpeg:

```sh
ffmpeg -i your_audio.mp3 -ar 16000 -ac 1 -c:a pcm_s16le output.wav
```
Then, update your Python script to use `output.wav`.

---

### **8️⃣ Run Everything Together**

```python
AUDIO_PATH = "your_audio_file.mp3"
transcript_text = transcribe_audio_whisper(AUDIO_PATH)

if transcript_text:
    find_movie_by_subtitle(transcript_text, top_k=10, similarity_threshold=0.5)
else:
    print("❌ No transcript generated. Check the audio file!")
```

---

## 🛠 Troubleshooting

### ❌ "Failed to load audio: FFmpeg error"
✔️ **Solution**: Reinstall FFmpeg and convert the audio format (Step 7).

### ❌ "No matching subtitles found!"
✔️ **Solution**: Increase `top_k` or lower `similarity_threshold` in `find_movie_by_subtitle()`.

---

## 📜 License
MIT License.
```
