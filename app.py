"""
HamDard - AI Mental Health Companion
Main Streamlit Application
Architecture: Gemini LLM + RAG + Emotion Detection + Crisis Safety
"""

import streamlit as st
from chatbot_engine import (
    create_chat_session,
    generate_response,
    generate_goodbye_summary,
)
from rag_pipeline import (
    get_chroma_client,
    get_or_create_collection,
    process_pdf,
    process_text_file,
    process_csv_conversations,
    add_document_to_kb,
    add_conversations_to_kb,
    get_kb_stats,
    clear_knowledge_base,
    initialize_default_knowledge,
)
from emotion_detector import detect_emotion

# =============================================
# PAGE CONFIG
# =============================================
st.set_page_config(
    page_title="HamDard - Mental Health Companion",
    page_icon="💚",
    layout="wide",
    initial_sidebar_state="expanded",
)

# =============================================
# CUSTOM CSS
# =============================================
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Nunito:wght@400;600;700;800&display=swap');

    /* Global */
    .stApp {
        font-family: 'Nunito', sans-serif;
    }

    /* Header */
    .hamdard-header {
        text-align: center;
        padding: 1.5rem 0 1rem 0;
    }
    .hamdard-header h1 {
        font-family: 'Nunito', sans-serif;
        font-size: 2.5rem;
        font-weight: 800;
        color: #2E7D32;
        margin: 0;
        letter-spacing: -0.5px;
    }
    .hamdard-header p {
        font-size: 1rem;
        color: #666;
        margin: 0.25rem 0 0 0;
    }

    /* Emotion Badge */
    .emotion-badge {
        display: inline-block;
        padding: 4px 14px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        margin: 4px 0;
    }
    .emotion-sadness { background: #E3F2FD; color: #1565C0; }
    .emotion-anxiety { background: #FFF3E0; color: #E65100; }
    .emotion-stress { background: #FCE4EC; color: #C62828; }
    .emotion-anger { background: #FFEBEE; color: #B71C1C; }
    .emotion-loneliness { background: #E8EAF6; color: #283593; }
    .emotion-frustration { background: #FFF8E1; color: #F57F17; }
    .emotion-hopelessness { background: #F3E5F5; color: #6A1B9A; }
    .emotion-confusion { background: #E0F7FA; color: #00695C; }
    .emotion-neutral { background: #E8F5E9; color: #2E7D32; }
    .emotion-crisis { background: #FFCDD2; color: #B71C1C; font-weight: 800; }

    /* Sidebar styling */
    .kb-stat {
        background: #F1F8E9;
        border-radius: 10px;
        padding: 12px 16px;
        margin: 8px 0;
        text-align: center;
    }
    .kb-stat h3 {
        margin: 0;
        font-size: 1.5rem;
        color: #2E7D32;
    }
    .kb-stat p {
        margin: 0;
        font-size: 0.85rem;
        color: #555;
    }

    /* Crisis banner */
    .crisis-banner {
        background: linear-gradient(135deg, #FFCDD2, #EF9A9A);
        border-radius: 10px;
        padding: 16px;
        margin: 10px 0;
        text-align: center;
        border: 1px solid #E57373;
    }
    .crisis-banner h4 {
        color: #B71C1C;
        margin: 0 0 8px 0;
    }
    .crisis-banner p {
        color: #C62828;
        margin: 2px 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# =============================================
# INITIALIZATION
# =============================================

def init_session_state():
    """Initialize all session state variables."""
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "conversation_emotions" not in st.session_state:
        st.session_state.conversation_emotions = []
    if "chat_session" not in st.session_state:
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            st.session_state.chat_session = create_chat_session(api_key)
        else:
            st.session_state.chat_session = None
    if "chroma_client" not in st.session_state:
        st.session_state.chroma_client = get_chroma_client()
    if "collection" not in st.session_state:
        st.session_state.collection = get_or_create_collection(
            st.session_state.chroma_client
        )
        # Load default knowledge
        initialize_default_knowledge(st.session_state.collection)


init_session_state()


# =============================================
# SIDEBAR
# =============================================

with st.sidebar:
    # --- About Section ---
    st.markdown("## 💚 HamDard")
    st.markdown(
        "An AI-powered mental health companion for Pakistani students, "
        "built with **Gemini LLM**, **RAG architecture**, and **emotion detection**."
    )

    st.divider()

    # --- Knowledge Base Stats ---
    st.markdown("### 📚 Knowledge Base")
    stats = get_kb_stats(st.session_state.collection)

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            f'<div class="kb-stat"><h3>{stats["total_chunks"]}</h3>'
            f'<p>Documents</p></div>',
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            f'<div class="kb-stat"><h3>{stats["num_sources"]}</h3>'
            f'<p>Sources</p></div>',
            unsafe_allow_html=True,
        )

    # --- Upload Documents ---
    st.markdown("### 📤 Upload to Knowledge Base")
    st.caption("Upload PDFs, text files, or CSV conversation datasets")

    uploaded_file = st.file_uploader(
        "Choose a file",
        type=["pdf", "txt", "csv"],
        help="PDF: documents, articles | TXT: plain text | CSV: patient-therapist conversations",
    )

    if uploaded_file is not None:
        if st.button("Add to Knowledge Base", type="primary", use_container_width=True):
            with st.spinner("Processing and adding to knowledge base..."):
                file_type = uploaded_file.name.split(".")[-1].lower()
                chunks_added = 0

                if file_type == "pdf":
                    text = process_pdf(uploaded_file, uploaded_file.name)
                    if text:
                        chunks_added = add_document_to_kb(
                            text, uploaded_file.name, st.session_state.collection
                        )

                elif file_type == "txt":
                    text = process_text_file(uploaded_file, uploaded_file.name)
                    if text:
                        chunks_added = add_document_to_kb(
                            text, uploaded_file.name, st.session_state.collection
                        )

                elif file_type == "csv":
                    conversations = process_csv_conversations(
                        uploaded_file, uploaded_file.name
                    )
                    if conversations:
                        chunks_added = add_conversations_to_kb(
                            conversations,
                            uploaded_file.name,
                            st.session_state.collection,
                        )

                if chunks_added > 0:
                    st.success(f"Added {chunks_added} chunks from {uploaded_file.name}")
                    st.rerun()
                else:
                    st.error("Could not extract content from the file.")

    # --- Clear KB ---
    if stats["total_chunks"] > 0:
        if st.button("Clear Knowledge Base", use_container_width=True):
            clear_knowledge_base(st.session_state.collection)
            # Re-add default knowledge
            initialize_default_knowledge(st.session_state.collection)
            st.success("Knowledge base reset to defaults.")
            st.rerun()

    st.divider()

    # --- Emotion Tracking ---
    st.markdown("### 🎭 Emotion Tracking")
    if st.session_state.conversation_emotions:
        emotions = st.session_state.conversation_emotions
        # Show last 5 emotions
        recent = emotions[-5:]
        for emo in recent:
            css_class = f"emotion-{emo}"
            st.markdown(
                f'<span class="emotion-badge {css_class}">{emo.capitalize()}</span>',
                unsafe_allow_html=True,
            )
    else:
        st.caption("Emotions will appear here as you chat.")

    st.divider()

    # --- Emergency Helplines ---
    st.markdown(
        '<div class="crisis-banner">'
        "<h4>Emergency Helplines</h4>"
        "<p><strong>Umang:</strong> 0311-7786264</p>"
        "<p><strong>Rozan:</strong> 0800-22444</p>"
        "<p><strong>Mental Health:</strong> 0800-00-009</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.divider()

    # --- New Conversation ---
    if st.button("🔄 New Conversation", use_container_width=True):
        # Show mood summary before clearing
        if st.session_state.conversation_emotions:
            summary = generate_goodbye_summary(
                st.session_state.conversation_emotions
            )
            st.info(summary)

        st.session_state.messages = []
        st.session_state.conversation_emotions = []
        api_key = st.secrets.get("GEMINI_API_KEY", "")
        if api_key:
            st.session_state.chat_session = create_chat_session(api_key)
        st.rerun()


# =============================================
# MAIN CHAT AREA
# =============================================

# Header
st.markdown(
    '<div class="hamdard-header">'
    "<h1>💚 HamDard</h1>"
    "<p>Your compassionate AI mental health companion | English • Urdu • Roman Urdu</p>"
    "</div>",
    unsafe_allow_html=True,
)

# Check API key
if not st.session_state.chat_session:
    st.error(
        "Gemini API key not found. Please add GEMINI_API_KEY to your Streamlit secrets."
    )
    st.stop()

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Show emotion badge for user messages
        if message["role"] == "user" and "emotion" in message:
            emo = message["emotion"]
            css_class = f"emotion-{emo}"
            st.markdown(
                f'<span class="emotion-badge {css_class}">Detected: {emo.capitalize()}</span>',
                unsafe_allow_html=True,
            )

# Welcome message
if not st.session_state.messages:
    welcome = (
        "Assalam-o-Alaikum! 💚 Main **HamDard** hoon, aapka digital companion.\n\n"
        "Aap mujhse **English**, **Urdu**, ya **Roman Urdu** mein baat kar sakte hain. "
        "Jo bhi aap ke dil mein ho, yahan share kar sakte hain — main sunne ke liye "
        "hamesha yahan hoon.\n\n"
        "**Aaj aap kaisa mehsoos kar rahe hain?**"
    )
    with st.chat_message("assistant"):
        st.markdown(welcome)
    st.session_state.messages.append({"role": "assistant", "content": welcome})

# Handle user input
if user_input := st.chat_input("Apni baat yahan likhein... | Type here..."):
    # Display user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Check for goodbye
    goodbye_words = [
        "bye", "goodbye", "allah hafiz", "khuda hafiz", "alvida",
        "good night", "take care", "shukriya",
    ]
    is_goodbye = any(word in user_input.lower() for word in goodbye_words)

    # Generate response
    with st.chat_message("assistant"):
        with st.spinner("Soch raha hoon..."):
            response_text, emotion_result = generate_response(
                user_input,
                st.session_state.chat_session,
                st.session_state.collection,
                st.session_state.conversation_emotions,
            )

            # Display response
            st.markdown(response_text)

            # If goodbye, show mood summary
            if is_goodbye and len(st.session_state.conversation_emotions) > 1:
                summary = generate_goodbye_summary(
                    st.session_state.conversation_emotions
                )
                st.markdown(f"\n\n---\n📊 **Mood Summary:** {summary}")
                response_text += f"\n\n---\n📊 **Mood Summary:** {summary}"

    # Update emotion in user message
    st.session_state.messages[-1]["emotion"] = emotion_result["emotion"]

    # Store assistant response
    st.session_state.messages.append(
        {"role": "assistant", "content": response_text}
    )

    st.rerun()
