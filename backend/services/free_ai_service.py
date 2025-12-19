"""
Free AI Service - Google Gemini, Edge TTS & Local Whisper Integration
Coach Ivy: Your personal English companion (Free Version)
"""
import logging
import google.generativeai as genai
import edge_tts
import tempfile
import os
import asyncio
from pathlib import Path
import hashlib
from fastapi import UploadFile
from config import settings
from faster_whisper import WhisperModel

logger = logging.getLogger(__name__)

# Configure Gemini
if settings.gemini_api_key:
    genai.configure(api_key=settings.gemini_api_key)
    model = genai.GenerativeModel(settings.gemini_model_name)
else:
    logger.warning("Gemini API Key not found. Chat features will not work.")
    model = None

# Initialize Whisper Model (Lazy loading to save startup time)
whisper_model = None

def get_whisper_model():
    global whisper_model
    if whisper_model is None:
        logger.info("Loading Whisper model (tiny)...")
        # 'tiny' is fast and good enough for clear speech. 'base' or 'small' are better but slower.
        whisper_model = WhisperModel("tiny", device="cpu", compute_type="int8")
    return whisper_model

# ===== COACH IVY SYSTEM PROMPTS =====
# (Reusing prompts from openai_service for consistency)

COACH_IVY_BASE_PROMPT = """You are "Coach Ivy", a personal English tutor for a Vietnamese learner.

**Personality:**
- Friendly, encouraging, and supportive
- Patient and understanding
- Enthusiastic about progress
- Professional but warm

**Communication style:**
- Use English for teaching and examples
- Use short Vietnamese explanations when concepts are complex
- Keep responses concise (2-4 sentences for most cases)
- Always be positive and motivating

**Teaching approach:**
- Focus on practical usage
- Provide real-world examples
- Explain "why" not just "what"
- Encourage practice and repetition
- Celebrate small wins
"""

MODE_PROMPTS = {
    "free_chat": """The user is having a casual conversation to practice English.
- Answer their questions naturally
- Gently correct major errors
- Keep the conversation flowing
- Use this as a teaching opportunity when appropriate""",

    "explain": """The user needs help understanding an English concept, word, or phrase.
- Provide a clear explanation
- Give 2-3 practical examples
- Include Vietnamese translation for key terms
- Keep it simple and actionable""",

    "speaking_feedback": """The user just practiced speaking. Provide constructive feedback.
- Start with encouragement
- Point out what they did well
- Suggest ONE main improvement
- Provide the corrected version
- Give a similar example to practice"""
}


def get_system_prompt(mode: str = "free_chat") -> str:
    """Get complete system prompt for Coach Ivy"""
    mode_specific = MODE_PROMPTS.get(mode, MODE_PROMPTS["free_chat"])
    return f"{COACH_IVY_BASE_PROMPT}\n\n{mode_specific}"


# ===== GEMINI FUNCTIONS =====

async def chat_with_coach(
    message: str,
    mode: str = "free_chat",
    context: dict = None
) -> tuple[str, str]:
    """
    Chat with Coach Ivy using Gemini
    """
    try:
        if not model:
            return "Error: Gemini API Key missing.", "neutral"

        system_prompt = get_system_prompt(mode)

        if context:
            system_prompt += f"\n\nContext: {context}"

        # Gemini doesn't have "system" role in the same way as GPT,
        # but we can prepend instructions.
        full_prompt = f"{system_prompt}\n\nUser: {message}"

        response = await model.generate_content_async(full_prompt)
        reply = response.text.strip()

        emotion_tag = _analyze_emotion(reply)
        logger.info(f"Coach Ivy (Gemini) replied (mode={mode}, emotion={emotion_tag})")

        return reply, emotion_tag

    except Exception as e:
        logger.error(f"Error in chat_with_coach (Gemini): {e}")
        return "Sorry, I'm having trouble thinking right now.", "neutral"


def _analyze_emotion(text: str) -> str:
    """Analyze text to determine appropriate emotion tag"""
    text_lower = text.lower()

    praise_words = ["excellent", "perfect", "great", "wonderful", "amazing", "fantastic", "correct", "well done", "good job", "tuyệt vời", "hoàn hảo"]
    if any(word in text_lower for word in praise_words):
        return "praise"

    corrective_words = ["however", "but", "correction", "should be", "mistake", "error", "incorrect", "sửa", "sai"]
    if any(word in text_lower for word in corrective_words):
        return "corrective"

    encouraging_words = ["keep", "practice", "try", "don't worry", "no problem", "keep going", "tiếp tục", "cố lên"]
    if any(word in text_lower for word in encouraging_words):
        return "encouraging"

    return "neutral"


# ===== EXERCISE FEEDBACK =====

async def check_exercise_with_feedback(
    question: str,
    user_answers: list[str],
    correct_answers: list[str],
    exercise_type: str = "multiple_choice"
) -> tuple[bool, float, str, str]:
    """Check exercise and generate AI feedback using Gemini"""
    try:
        is_correct = user_answers == correct_answers
        score = 100.0 if is_correct else 0.0

        prompt = f"""{get_system_prompt("explain")}

The student answered this question:
Question: {question}
Their answer: {' '.join(user_answers)}
Correct answer: {' '.join(correct_answers)}

Provide brief feedback (2-3 sentences):
- If correct: praise and explain why it's right
- If incorrect: gently explain the mistake and provide the correct answer with reasoning"""

        if not model:
             return is_correct, score, "Feedback unavailable (No API Key)", "neutral"

        response = await model.generate_content_async(prompt)
        feedback = response.text.strip()
        emotion_tag = "praise" if is_correct else "corrective"

        return is_correct, score, feedback, emotion_tag

    except Exception as e:
        logger.error(f"Error in check_exercise_with_feedback (Gemini): {e}")
        return is_correct, score, "Good job!", "neutral"


# ===== EDGE TTS FUNCTIONS =====

MEDIA_DIR = Path("media/tts")
MEDIA_DIR.mkdir(parents=True, exist_ok=True)

def _get_audio_hash(text: str, voice: str) -> str:
    content = f"{text}_{voice}"
    return hashlib.md5(content.encode()).hexdigest()

async def generate_speech(
    text: str,
    voice: str = None
) -> str:
    """Generate speech using Edge TTS (Free)"""
    try:
        # Default voice: en-US-AvaNeural (very natural female voice)
        voice = voice or "en-US-AvaNeural"

        audio_hash = _get_audio_hash(text, voice)
        audio_path = MEDIA_DIR / f"{audio_hash}.mp3"

        if audio_path.exists():
            return str(audio_path)

        logger.info(f"Generating Edge TTS for: {text[:50]}...")
        communicate = edge_tts.Communicate(text, voice)
        await communicate.save(str(audio_path))

        return str(audio_path)

    except Exception as e:
        logger.error(f"Error in generate_speech (EdgeTTS): {e}")
        raise


# ===== WHISPER (LOCAL) FUNCTIONS =====

async def transcribe_audio(
    file: UploadFile,
    language: str = "en"
) -> str:
    """Transcribe audio using local Faster Whisper"""
    temp_file_path = None
    try:
        file_extension = Path(file.filename).suffix or ".webm"
        with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as temp_file:
            temp_file_path = temp_file.name
            content = await file.read()
            temp_file.write(content)
            temp_file.flush()

        logger.info(f"Transcribing audio (Local Whisper): {file.filename}")

        # Run in thread pool to avoid blocking async loop
        loop = asyncio.get_event_loop()
        segments, _ = await loop.run_in_executor(
            None,
            lambda: get_whisper_model().transcribe(temp_file_path, language=language)
        )

        transcript = " ".join([segment.text for segment in segments]).strip()
        logger.info(f"Transcription complete: {transcript[:100]}...")

        return transcript

    except Exception as e:
        logger.error(f"Error in transcribe_audio (Local): {e}")
        raise

    finally:
        if temp_file_path and os.path.exists(temp_file_path):
            try:
                os.unlink(temp_file_path)
            except Exception:
                pass


# ===== BILINGUAL FEEDBACK =====

async def generate_bilingual_feedback(
    expected_text: str,
    spoken_text: str,
    word_accuracy: float,
    accuracy_details: dict
) -> tuple[str, str, list[str]]:
    """Generate bilingual feedback using Gemini"""
    try:
        matches = accuracy_details.get('matches', 0)
        substitutions = accuracy_details.get('substitutions', 0)
        deletions = accuracy_details.get('deletions', 0)
        insertions = accuracy_details.get('insertions', 0)

        prompt = f"""{get_system_prompt("speaking_feedback")}

The student practiced reading aloud in English.
Expected: "{expected_text}"
They said: "{spoken_text}"
Accuracy: {word_accuracy:.1f}% (Correct: {matches}, Wrong: {substitutions}, Missing: {deletions})

Generate JSON:
{{
  "feedback_en": "1-2 sentences English feedback",
  "feedback_vi": "1-2 sentences Vietnamese feedback",
  "tricky_words": ["word1", "word2"]
}}"""

        if not model:
            return "Good practice!", "Luyện tập tốt!", []

        response = await model.generate_content_async(prompt)
        response_text = response.text.strip()

        # Parse JSON (basic cleanup)
        import json
        if "```json" in response_text:
            response_text = response_text.split("```json")[1].split("```")[0].strip()
        elif "```" in response_text:
            response_text = response_text.split("```")[1].split("```")[0].strip()

        data = json.loads(response_text)
        return (
            data.get("feedback_en", "Good job!"),
            data.get("feedback_vi", "Tốt lắm!"),
            data.get("tricky_words", [])
        )

    except Exception as e:
        logger.error(f"Error in bilingual feedback (Gemini): {e}")
        return "Keep practicing!", "Tiếp tục cố gắng nhé!", []
