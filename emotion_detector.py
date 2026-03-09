"""
Emotion Detection Module for HamDard Chatbot
Detects emotions from user text in English, Urdu, and Roman Urdu
"""

# Emotion keywords mapping - English, Roman Urdu, and Urdu
EMOTION_KEYWORDS = {
    "sadness": {
        "keywords": [
            "sad", "unhappy", "depressed", "crying", "tears", "heartbroken", "miserable",
            "hopeless", "empty", "lonely", "alone", "lost", "hurt", "pain", "broken",
            "dukhi", "udaas", "rona", "ro raha", "ro rahi", "dil toota", "tanha",
            "akela", "akeli", "takleef", "dard", "toot gaya", "koi nahi hai",
            "zindagi mushkil", "bohat bura", "dil nahi lagta"
        ],
        "response_tone": "gentle, warm, validating"
    },
    "anxiety": {
        "keywords": [
            "anxious", "worried", "nervous", "panic", "scared", "fear", "overthinking",
            "restless", "uneasy", "tense", "stressed", "cant sleep", "racing thoughts",
            "tension", "ghabra", "ghabrahat", "dar", "darr", "khauf", "pareshan",
            "neend nahi", "soch soch ke", "kya hoga", "fikar", "bechaini",
            "dil ghabrata", "phat rahi hai"
        ],
        "response_tone": "calming, reassuring, grounding"
    },
    "stress": {
        "keywords": [
            "stressed", "overwhelmed", "pressure", "too much", "cant handle", "burnout",
            "exhausted", "tired", "workload", "deadline", "exams", "studies",
            "tension", "bohat zyada", "bardasht nahi", "thak gaya", "thak gayi",
            "dimagh kharab", "sar dard", "kaam bohat", "imtihan", "parhai",
            "pressure hai", "handle nahi ho raha", "pagal ho jaunga"
        ],
        "response_tone": "supportive, practical, encouraging"
    },
    "anger": {
        "keywords": [
            "angry", "furious", "mad", "hate", "frustrated", "annoyed", "irritated",
            "rage", "unfair", "sick of", "fed up",
            "gussa", "ghussa", "nafrat", "tang", "bezaar", "chir", "jal raha",
            "insaaf nahi", "tang aa gaya", "tang aa gayi", "sab se nafrat"
        ],
        "response_tone": "validating, calm, non-judgmental"
    },
    "loneliness": {
        "keywords": [
            "lonely", "alone", "no friends", "no one cares", "isolated", "left out",
            "abandoned", "rejected", "nobody", "invisible",
            "akela", "akeli", "tanha", "koi nahi", "kisi ko parwa nahi",
            "sab door", "koi dost nahi", "koi samajhta nahi", "miss kar raha"
        ],
        "response_tone": "warm, connecting, companionable"
    },
    "frustration": {
        "keywords": [
            "frustrated", "stuck", "nothing works", "useless", "pointless", "why me",
            "cant do anything", "failing", "failure", "hopeless",
            "kuch nahi hota", "bekar", "fail", "nakaam", "kya karun",
            "samajh nahi aata", "haar gaya", "haar gayi", "kuch nahi ban sakta"
        ],
        "response_tone": "encouraging, solution-oriented, empathetic"
    },
    "hopelessness": {
        "keywords": [
            "hopeless", "no point", "give up", "cant go on", "whats the point",
            "nothing matters", "no future", "no hope", "end it",
            "umeed nahi", "koi faida nahi", "chor do", "khatam", "matlab nahi",
            "aage kuch nahi", "zindagi bekar", "sab khatam"
        ],
        "response_tone": "gentle, hopeful, crisis-aware"
    },
    "confusion": {
        "keywords": [
            "confused", "dont know", "lost", "unsure", "what should i do",
            "no idea", "help me", "cant decide", "mixed feelings",
            "samajh nahi", "pata nahi", "kya karun", "confuse", "faisla nahi",
            "madad karo", "kuch samajh nahi aa raha"
        ],
        "response_tone": "clarifying, patient, guiding"
    }
}

# Crisis keywords - HIGH PRIORITY detection
CRISIS_KEYWORDS = [
    "suicide", "suicidal", "kill myself", "want to die", "end my life",
    "self harm", "self-harm", "cut myself", "cutting", "hurt myself",
    "no reason to live", "better off dead", "overdose", "jump off",
    "marna chahta", "marna chahti", "mar jaunga", "mar jaungi",
    "khudkushi", "zindagi khatam", "jeena nahi", "maut chahiye",
    "apne aap ko hurt", "kaat", "suicide kar lunga", "mar jana chahta",
    "life ka koi matlab nahi", "sab khatam kar dunga", "mujhe nahi jeena"
]


def detect_emotion(text):
    """
    Detect the primary emotion from user text.
    Returns a dict with emotion, confidence, is_crisis, and response_tone.
    """
    text_lower = text.lower().strip()

    # --- Crisis Detection (HIGHEST PRIORITY) ---
    for keyword in CRISIS_KEYWORDS:
        if keyword in text_lower:
            return {
                "emotion": "crisis",
                "confidence": 1.0,
                "is_crisis": True,
                "response_tone": "immediate care, provide helplines, express deep concern",
                "detected_keywords": [keyword]
            }

    # --- Emotion Detection ---
    emotion_scores = {}
    detected_keywords_map = {}

    for emotion, data in EMOTION_KEYWORDS.items():
        score = 0
        found_keywords = []
        for keyword in data["keywords"]:
            if keyword in text_lower:
                score += 1
                found_keywords.append(keyword)
        if score > 0:
            emotion_scores[emotion] = score
            detected_keywords_map[emotion] = found_keywords

    # If no emotion detected, return neutral
    if not emotion_scores:
        return {
            "emotion": "neutral",
            "confidence": 0.5,
            "is_crisis": False,
            "response_tone": "friendly, conversational, warm",
            "detected_keywords": []
        }

    # Get the highest scoring emotion
    primary_emotion = max(emotion_scores, key=emotion_scores.get)
    max_score = emotion_scores[primary_emotion]

    # Simple confidence: based on number of keyword matches
    confidence = min(max_score / 3.0, 1.0)

    return {
        "emotion": primary_emotion,
        "confidence": round(confidence, 2),
        "is_crisis": False,
        "response_tone": EMOTION_KEYWORDS[primary_emotion]["response_tone"],
        "detected_keywords": detected_keywords_map.get(primary_emotion, [])
    }


def get_emotion_summary(emotions_list):
    """
    Generate a mood summary from a list of detected emotions in a conversation.
    Returns a summary string.
    """
    if not emotions_list:
        return "No mood data available for this session."

    # Filter out neutral
    meaningful = [e for e in emotions_list if e != "neutral"]

    if not meaningful:
        return "You seemed to be in a calm and neutral mood today. That's perfectly okay!"

    # Count occurrences
    from collections import Counter
    counts = Counter(meaningful)
    primary = counts.most_common(1)[0][0]

    summary_map = {
        "sadness": "You seemed to be carrying some sadness today. Remember, it's okay to feel this way, and reaching out was a brave step.",
        "anxiety": "You appeared to be feeling anxious today. Remember to breathe, and know that this feeling will pass.",
        "stress": "You seemed quite stressed today. Please take some time to rest and be kind to yourself.",
        "anger": "You seemed frustrated or angry today. Your feelings are valid. Try to find a healthy outlet.",
        "loneliness": "You seemed to be feeling lonely today. Remember, you matter and you're not as alone as you might feel.",
        "frustration": "You seemed frustrated today. Setbacks are temporary. You're stronger than you think.",
        "hopelessness": "You seemed to be going through a really tough time. Please remember there is always hope, even when it's hard to see.",
        "confusion": "You seemed unsure about some things today. That's completely normal. Take it one step at a time.",
        "crisis": "You shared some very heavy feelings today. Please reach out to a professional or helpline. You deserve support."
    }

    return summary_map.get(primary, f"You seemed to be feeling {primary} today. Take care of yourself!")
