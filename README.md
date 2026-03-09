# 💚 HamDard - AI Mental Health Companion

An AI-powered mental health chatbot designed for Pakistani students and young people. Built using **RAG (Retrieval-Augmented Generation)** architecture with **Google Gemini LLM**, **emotion detection**, and **multilingual support** (English, Urdu, Roman Urdu).

## Architecture

```
User Message
    │
    ▼
┌─────────────────┐
│ Emotion Detector │──→ Detects: sadness, anxiety, stress, anger,
│ (Rule-based NLP) │    loneliness, frustration, hopelessness,
└────────┬────────┘    confusion, crisis, neutral
         │
         ▼
┌─────────────────┐
│  RAG Pipeline   │──→ Queries ChromaDB vector store
│ (ChromaDB +     │    Retrieves relevant knowledge chunks
│  Embeddings)    │    (coping techniques, therapy patterns)
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Gemini LLM     │──→ System prompt + Emotion context
│ (gemini-2.5-    │    + RAG context + User message
│  flash)         │    = Empathetic, contextual response
└────────┬────────┘
         │
         ▼
    Response + Emotion Badge + Mood Tracking
```

## Features

- **LLM Integration**: Google Gemini 2.5 Flash for natural, empathetic responses
- **RAG Knowledge Base**: ChromaDB vector store with uploadable PDFs, TXT, and CSV files
- **Emotion Detection**: Rule-based NLP module detecting 9 emotions + crisis state
- **Multilingual**: English, Urdu, and Roman Urdu support
- **Crisis Safety**: Automatic detection of self-harm/suicidal content with helpline info
- **Mood Tracking**: Session-based emotion tracking with summary
- **Conversation Memory**: Gemini chat session maintains full conversation context
- **Expandable KB**: Upload patient-therapist datasets (CSV) to improve responses

## Project Structure

```
hamdard-chatbot/
├── app.py                 # Main Streamlit application (UI + routing)
├── chatbot_engine.py      # Core engine (LLM + RAG + Emotion integration)
├── emotion_detector.py    # Emotion detection module
├── rag_pipeline.py        # RAG pipeline (ChromaDB, document processing)
├── requirements.txt       # Python dependencies
├── knowledge_base/        # Folder for local documents (optional)
└── README.md              # This file
```

## Setup

1. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up Gemini API key:**
   - Get a free key from https://aistudio.google.com/apikey
   - Create `.streamlit/secrets.toml`:
     ```toml
     GEMINI_API_KEY = "your-api-key-here"
     ```

3. **Run the app:**
   ```bash
   streamlit run app.py
   ```

## Adding Your Dataset (RAG Knowledge Base)

You can expand the chatbot's knowledge by uploading files through the sidebar:

- **PDF**: Mental health articles, therapy guides, research papers
- **TXT**: Plain text documents, notes
- **CSV**: Patient-therapist conversation datasets
  - Expected columns: `patient` (or `user`/`input`) and `therapist` (or `assistant`/`response`)

The RAG pipeline will automatically chunk, embed, and store the documents in ChromaDB.

## Deployment (Streamlit Cloud)

1. Push code to GitHub
2. Go to https://share.streamlit.io
3. Connect your repo
4. Add `GEMINI_API_KEY` in Advanced Settings → Secrets
5. Deploy

## Tech Stack

| Component | Technology |
|-----------|-----------|
| LLM | Google Gemini 2.5 Flash |
| Vector Database | ChromaDB |
| RAG Framework | LangChain Text Splitters |
| Emotion Detection | Custom Rule-based NLP |
| Frontend | Streamlit |
| Document Processing | PyPDF2, Pandas |

## Emergency Helplines (Pakistan)

- **Umang Helpline**: 0311-7786264
- **Rozan Counseling**: 0800-22444
- **Mental Health Helpline**: 0800-00-009

## Disclaimer

HamDard is NOT a replacement for professional mental health therapy. It is a supportive digital companion designed to provide emotional support and coping strategies. If you are in crisis, please contact a professional helpline immediately.
