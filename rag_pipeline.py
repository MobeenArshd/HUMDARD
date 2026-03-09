"""
RAG (Retrieval-Augmented Generation) Pipeline for HamDard Chatbot
Uses ChromaDB for vector storage and sentence-transformers for embeddings.
Allows uploading PDFs, text files, and CSV datasets as knowledge base.
"""

import os
import json
import hashlib
import chromadb
from chromadb.config import Settings
from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2
import pandas as pd

# --- Configuration ---
CHROMA_DB_PATH = "./chroma_db"
COLLECTION_NAME = "hamdard_knowledge"
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50


def get_chroma_client():
    """Initialize and return ChromaDB client."""
    client = chromadb.PersistentClient(path=CHROMA_DB_PATH)
    return client


def get_or_create_collection(client):
    """Get or create the knowledge base collection."""
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"description": "HamDard mental health knowledge base"}
    )
    return collection


def generate_doc_id(text, source="unknown"):
    """Generate a unique document ID based on content hash."""
    content = f"{source}:{text}"
    return hashlib.md5(content.encode()).hexdigest()


def split_text_into_chunks(text, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):
    """Split text into smaller chunks for embedding."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    chunks = splitter.split_text(text)
    return chunks


# =============================================
# DOCUMENT PROCESSING FUNCTIONS
# =============================================

def process_pdf(file_path_or_bytes, source_name="uploaded_pdf"):
    """
    Extract text from a PDF file.
    Accepts file path (str) or file-like object (bytes).
    """
    text = ""
    try:
        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, "rb") as f:
                reader = PyPDF2.PdfReader(f)
                for page in reader.pages:
                    page_text = page.extract_text()
                    if page_text:
                        text += page_text + "\n"
        else:
            reader = PyPDF2.PdfReader(file_path_or_bytes)
            for page in reader.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        print(f"Error reading PDF: {e}")
        return ""
    return text.strip()


def process_text_file(file_path_or_bytes, source_name="uploaded_txt"):
    """Extract text from a .txt file."""
    try:
        if isinstance(file_path_or_bytes, str):
            with open(file_path_or_bytes, "r", encoding="utf-8") as f:
                return f.read().strip()
        else:
            return file_path_or_bytes.read().decode("utf-8").strip()
    except Exception as e:
        print(f"Error reading text file: {e}")
        return ""


def process_csv_conversations(file_path_or_bytes, source_name="uploaded_csv"):
    """
    Process CSV file containing patient-therapist conversations.
    Expected columns: 'patient' and 'therapist' (or similar).
    Converts each row into a formatted conversation chunk.
    """
    try:
        if isinstance(file_path_or_bytes, str):
            df = pd.read_csv(file_path_or_bytes)
        else:
            df = pd.read_csv(file_path_or_bytes)

        conversations = []
        columns_lower = [c.lower() for c in df.columns]

        # Try to detect patient/therapist columns
        patient_col = None
        therapist_col = None

        for i, col in enumerate(columns_lower):
            if any(word in col for word in ["patient", "user", "client", "input", "question"]):
                patient_col = df.columns[i]
            if any(word in col for word in ["therapist", "assistant", "counselor", "response", "answer", "output"]):
                therapist_col = df.columns[i]

        if patient_col and therapist_col:
            for _, row in df.iterrows():
                conv = f"Patient: {row[patient_col]}\nTherapist: {row[therapist_col]}"
                conversations.append(conv)
        else:
            # If columns not detected, concatenate all text columns
            for _, row in df.iterrows():
                text_parts = [str(val) for val in row.values if pd.notna(val)]
                conversations.append(" ".join(text_parts))

        return conversations

    except Exception as e:
        print(f"Error reading CSV: {e}")
        return []


# =============================================
# KNOWLEDGE BASE MANAGEMENT
# =============================================

def add_document_to_kb(text, source_name, collection):
    """
    Add a document to the knowledge base.
    Splits into chunks, generates IDs, and stores in ChromaDB.
    Returns number of chunks added.
    """
    if not text or not text.strip():
        return 0

    chunks = split_text_into_chunks(text)
    if not chunks:
        return 0

    ids = []
    documents = []
    metadatas = []

    for i, chunk in enumerate(chunks):
        doc_id = generate_doc_id(chunk, source_name)
        ids.append(doc_id)
        documents.append(chunk)
        metadatas.append({
            "source": source_name,
            "chunk_index": i,
            "total_chunks": len(chunks)
        })

    # Add to ChromaDB (upsert to avoid duplicates)
    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return len(chunks)


def add_conversations_to_kb(conversations, source_name, collection):
    """
    Add patient-therapist conversations to the knowledge base.
    Each conversation is stored as a separate document.
    """
    if not conversations:
        return 0

    ids = []
    documents = []
    metadatas = []

    for i, conv in enumerate(conversations):
        doc_id = generate_doc_id(conv, source_name)
        ids.append(doc_id)
        documents.append(conv)
        metadatas.append({
            "source": source_name,
            "type": "conversation",
            "conversation_index": i
        })

    collection.upsert(
        ids=ids,
        documents=documents,
        metadatas=metadatas
    )

    return len(conversations)


def query_knowledge_base(query, collection, n_results=3):
    """
    Search the knowledge base for relevant documents.
    Returns a list of relevant text chunks.
    """
    try:
        count = collection.count()
        if count == 0:
            return []

        results = collection.query(
            query_texts=[query],
            n_results=min(n_results, count)
        )

        if results and results["documents"]:
            return results["documents"][0]  # First query's results
        return []

    except Exception as e:
        print(f"Error querying knowledge base: {e}")
        return []


def get_kb_stats(collection):
    """Get statistics about the knowledge base."""
    try:
        count = collection.count()
        # Get unique sources
        if count > 0:
            all_data = collection.get(include=["metadatas"])
            sources = set()
            for meta in all_data["metadatas"]:
                if meta and "source" in meta:
                    sources.add(meta["source"])
            return {
                "total_chunks": count,
                "sources": list(sources),
                "num_sources": len(sources)
            }
        return {"total_chunks": 0, "sources": [], "num_sources": 0}
    except Exception as e:
        print(f"Error getting KB stats: {e}")
        return {"total_chunks": 0, "sources": [], "num_sources": 0}


def clear_knowledge_base(collection):
    """Clear all documents from the knowledge base."""
    try:
        ids = collection.get()["ids"]
        if ids:
            collection.delete(ids=ids)
        return True
    except Exception as e:
        print(f"Error clearing KB: {e}")
        return False


# =============================================
# DEFAULT KNOWLEDGE BASE CONTENT
# =============================================

DEFAULT_MENTAL_HEALTH_KNOWLEDGE = [
    """Breathing Exercise - 4-7-8 Technique: This is a calming breathing technique. 
    Breathe in through your nose for 4 seconds. Hold your breath for 7 seconds. 
    Exhale slowly through your mouth for 8 seconds. Repeat 3-4 times. 
    This activates your parasympathetic nervous system and helps reduce anxiety.
    In Roman Urdu: 4 second saans andar lo, 7 second rok ke rakho, 8 second mein 
    dheere se bahar nikalo. Ye technique anxiety aur tension kam karti hai.""",

    """Grounding Technique - 5-4-3-2-1 Method: When feeling anxious or overwhelmed, 
    use your senses to ground yourself. Notice 5 things you can SEE, 4 things you can 
    TOUCH, 3 things you can HEAR, 2 things you can SMELL, 1 thing you can TASTE.
    Roman Urdu: Jab anxiety ho to 5 cheezein dekho, 4 cheezein chuo, 3 awazein suno, 
    2 cheezein sungho, 1 cheez ka taste lo. Ye aapko present moment mein laata hai.""",

    """Journaling for Mental Health: Writing down your thoughts and feelings can help 
    process emotions. Try these prompts: What am I feeling right now and why? 
    What are three things I am grateful for today? What would I tell a friend in my situation?
    Roman Urdu: Apne khayal likhna bohat madad karta hai. Likhein ke aaj aap kya mehsoos 
    kar rahe hain, 3 cheezein jin ka shukr hai, aur apne dost ko kya mashwara dete.""",

    """Progressive Muscle Relaxation: Tense each muscle group for 5 seconds, then release.
    Start from your toes, move to calves, thighs, stomach, hands, arms, shoulders, and face.
    This helps release physical tension caused by stress and anxiety.
    Roman Urdu: Apne muscles ko 5 second ke liye tight karein phir chor dein. 
    Paon se shuru karein aur face tak jayein. Ye tension door karta hai.""",

    """Sleep Hygiene Tips for Students: Maintain a consistent sleep schedule. Avoid screens 
    30 minutes before bed. Keep your room cool and dark. Avoid caffeine after 2 PM.
    Try relaxation techniques before sleeping. Good sleep is essential for mental health.
    Roman Urdu: Neend achi karne ke liye roz ek hi waqt soyen, phone band karein sone se 
    pehle, kamra thanda rakhen, chai dopehr ke baad na piyen.""",

    """When to Seek Professional Help: If you experience persistent sadness lasting more 
    than 2 weeks, inability to function in daily life, thoughts of self-harm or suicide, 
    severe anxiety that interferes with activities, or substance abuse, please seek 
    professional help. In Pakistan, you can contact: Umang Helpline (0311-7786264), 
    Rozan Counseling (0800-22444), or Pakistan Mental Health Helpline (0800-00-009).
    Roman Urdu: Agar 2 hafte se zyada udaasi ho, daily kaam na ho sake, khud ko hurt 
    karne ke khayal ayein, to please professional se madad lein.""",

    """Mindfulness Meditation for Beginners: Sit comfortably. Close your eyes. Focus on 
    your breathing. When your mind wanders, gently bring it back to your breath. Start 
    with 5 minutes daily and gradually increase. Mindfulness reduces stress and improves 
    emotional regulation.
    Roman Urdu: Aram se baithen, aankhein band karein, saans par dhyan den. Jab dimagh 
    bhatkay to wapis saans par layen. 5 minute se shuru karein. Ye stress kam karta hai.""",

    """Understanding Exam Stress: Exam stress is common among students. Symptoms include 
    difficulty sleeping, racing thoughts, difficulty concentrating, and physical symptoms 
    like headaches. Manage it by: creating a study schedule, taking regular breaks, 
    practicing relaxation, eating well, and talking to someone about your worries.
    Roman Urdu: Imtihan ka stress normal hai. Schedule banayein, breaks lein, relax karein, 
    acha khana khayein, aur kisi se apni tension share karein."""
]


def initialize_default_knowledge(collection):
    """Load default mental health knowledge into the knowledge base."""
    stats = get_kb_stats(collection)
    # Only add if KB is empty
    if stats["total_chunks"] == 0:
        for i, text in enumerate(DEFAULT_MENTAL_HEALTH_KNOWLEDGE):
            add_document_to_kb(text, f"default_knowledge_{i}", collection)
        return len(DEFAULT_MENTAL_HEALTH_KNOWLEDGE)
    return 0
