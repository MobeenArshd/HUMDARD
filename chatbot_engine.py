"""
HamDard Chatbot Engine
Integrates: Gemini LLM + RAG Pipeline + Emotion Detection + Safety Logic
"""

import google.generativeai as genai
from emotion_detector import detect_emotion, get_emotion_summary
from rag_pipeline import query_knowledge_base


# =============================================
# SYSTEM PROMPT
# =============================================

SYSTEM_PROMPT = (
    "You are HamDard — a compassionate, culturally-aware AI mental health "
    "companion designed for Pakistani students and young people.\n\n"

    "CORE IDENTITY:\n"
    "- You are NOT a therapist or doctor. You are a supportive digital companion.\n"
    "- You provide emotional support, empathetic listening, and coping suggestions.\n"
    "- You are warm, gentle, non-judgmental, and patient.\n"
    "- You remember what the user said earlier in the conversation and refer back to it naturally.\n\n"

    "LANGUAGE RULES:\n"
    "- You can understand and respond in English, Urdu, and Roman Urdu.\n"
    "- If the user writes in Roman Urdu (e.g., 'mujhe bohat tension ho rahi hai'), "
    "respond in Roman Urdu mixed with simple English.\n"
    "- If the user writes in English, respond in simple English.\n"
    "- Match the user's language naturally. Never force a language switch.\n\n"

    "RESPONSE GUIDELINES:\n"
    "1. ALWAYS start by acknowledging the user's feelings (validation first).\n"
    "2. Ask gentle follow-up questions to understand better.\n"
    "3. When you have relevant knowledge from the context provided, use it naturally "
    "in your response without mentioning that it came from a database.\n"
    "4. Keep responses concise (3-5 sentences usually). Don't lecture.\n"
    "5. Use a conversational, friendly tone — like a caring friend.\n"
    "6. Never start responses with 'I understand' every time — vary your openings.\n\n"

    "BOUNDARIES:\n"
    "- Never diagnose any mental health condition.\n"
    "- Never prescribe medication.\n"
    "- Never claim to replace professional therapy.\n"
    "- If asked medical questions, redirect to professionals.\n"
    "- Keep conversations appropriate and safe.\n"
)

CRISIS_PROMPT = (
    "\n\nCRISIS DETECTED - IMMEDIATE PROTOCOL:\n"
    "The user may be in emotional crisis. You MUST:\n"
    "1. Express deep, genuine care and concern.\n"
    "2. Tell them they are NOT alone and their feelings matter.\n"
    "3. Provide these Pakistani helpline numbers:\n"
    "   - Umang Helpline: 0311-7786264\n"
    "   - Rozan Counseling: 0800-22444\n"
    "   - Pakistan Mental Health Helpline: 0800-00-009\n"
    "4. Strongly encourage them to talk to a trusted person (family, friend, teacher).\n"
    "5. Do NOT try to be their therapist. Focus on immediate safety and connection.\n"
    "6. Be gentle and warm. No clinical language.\n"
)


def build_rag_context(user_message, collection):
    """
    Query the RAG knowledge base and build context for the LLM.
    """
    relevant_docs = query_knowledge_base(user_message, collection, n_results=3)

    if not relevant_docs:
        return ""

    context = "\n\nRELEVANT KNOWLEDGE (use naturally in your response, do not mention the source):\n"
    for i, doc in enumerate(relevant_docs, 1):
        context += f"---\n{doc}\n"
    context += "---\n"

    return context


def build_emotion_context(emotion_result):
    """
    Build emotion-aware instructions for the LLM based on detected emotion.
    """
    emotion = emotion_result["emotion"]
    tone = emotion_result["response_tone"]
    confidence = emotion_result["confidence"]

    context = f"\n\nDETECTED EMOTION: {emotion} (confidence: {confidence})\n"
    context += f"RECOMMENDED TONE: {tone}\n"
    context += "Tailor your response to address this emotional state naturally.\n"

    return context


def generate_response(user_message, chat_session, collection, conversation_emotions):
    """
    Main response generation function.
    1. Detect emotion
    2. Check for crisis
    3. Query RAG knowledge base
    4. Build enhanced prompt
    5. Generate LLM response
    6. Track emotions

    Returns: (response_text, emotion_result)
    """
    # Step 1: Detect emotion
    emotion_result = detect_emotion(user_message)
    conversation_emotions.append(emotion_result["emotion"])

    # Step 2: Build RAG context
    rag_context = build_rag_context(user_message, collection)

    # Step 3: Build emotion context
    emotion_context = build_emotion_context(emotion_result)

    # Step 4: Build the enhanced message
    enhanced_message = user_message

    # Add RAG context if available
    if rag_context:
        enhanced_message += rag_context

    # Add emotion context
    enhanced_message += emotion_context

    # Add crisis protocol if needed
    if emotion_result["is_crisis"]:
        enhanced_message += CRISIS_PROMPT

    # Step 5: Generate response using Gemini
    try:
        response = chat_session.send_message(enhanced_message)
        return response.text, emotion_result
    except Exception as e:
        error_msg = (
            "I'm sorry, I'm having trouble responding right now. "
            "If you're in distress, please contact:\n"
            "- Umang Helpline: 0311-7786264\n"
            "- Rozan Counseling: 0800-22444\n"
            "- Pakistan Mental Health Helpline: 0800-00-009"
        )
        return error_msg, emotion_result


def create_chat_session(api_key):
    """
    Initialize the Gemini model and create a chat session.
    """
    genai.configure(api_key=api_key)

    model = genai.GenerativeModel(
        model_name="gemini-2.5-flash",
        system_instruction=SYSTEM_PROMPT
    )

    chat_session = model.start_chat(history=[])
    return chat_session


def generate_goodbye_summary(conversation_emotions):
    """
    Generate a mood summary when user says goodbye.
    """
    return get_emotion_summary(conversation_emotions)
