# 🎬 AI-Powered Subtitle Search Engine  

## 📌 Overview

This project enables **AI-powered subtitle search** by matching video subtitles with user-uploaded **audio queries**. It leverages:

- **Google Gemini API** for text embeddings
- **Whisper ASR** for speech-to-text conversion
- **ChromaDB** for storing and retrieving subtitle embeddings
- **Streamlit** for a user-friendly interface

## 🚀 Features
✅ Upload an **audio clip (MP3/WAV/M4A)**
✅ Convert speech to text using **Whisper ASR**
✅ Generate **semantic embeddings** using **Google Gemini API**
✅ Retrieve and rank **matching subtitles** using **cosine similarity**
✅ Display the **top relevant subtitles**

## 🛠️ Installation

### 1️⃣ Clone the Repository

```sh
git clone https://github.com/your-repo/subtitle-search-engine.git
cd subtitle-search-engine
```

### 2️⃣ Install Dependencies

```sh
pip install -r requirements.txt
```

### 3️⃣ Set Up Google Gemini API

Get your **Google Gemini API key** and add it to your environment variables:

```sh
export GOOGLE_API_KEY="your_api_key_here"
```

### 4️⃣ Run the Application

```sh
streamlit run app.py
```

## 📂 Project Structure

```
📁 subtitle-search-engine
│── 📄 app.py                 # Main Streamlit app
│── 📄 requirements.txt       # Required dependencies
│── 📄 README.md              # Project documentation
│── 📁 data                   # Subtitle database files
│── 📁 embeddings             # ChromaDB embeddings
```

## 🔥 Future Enhancements

- 🎭 **Multi-language support**  
- 🎞️ **Real-time video subtitle matching**  
- 📡 **Cloud-based deployment**  

## 📜 License  

This project is open-source under the **MIT License**.  

🚀 **Contributions Welcome!** Feel free to fork and improve the project!
