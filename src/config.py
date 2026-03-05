"""Central configuration for voicebot pipeline."""
import os
import logging

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("voicebot")

# HuggingFace
HF_TOKEN = os.getenv("HF_TOKEN")

# Model IDs
WHISPER_MODEL = os.getenv("WHISPER_MODEL", "openai/whisper-large-v3")
QWEN_MODEL = os.getenv("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
TTS_MODEL = os.getenv("TTS_MODEL", "DragonLineageAI/Vi-SparkTTS-0.5B")

# Device
DEVICE = "cuda"

# STT config
STT_SAMPLE_RATE = 16000
STT_LANGUAGE = "vi"

# LLM config
LLM_MAX_NEW_TOKENS = 96
LLM_TEMPERATURE = 0.7
DEFAULT_SYSTEM_PROMPT = (
    "Bạn là trợ lý AI. LUÔN trả lời bằng tiếng Việt. "
    "Trả lời ngắn gọn, tự nhiên, phù hợp cho hội thoại giọng nói. "
    "Tối đa 2 câu mỗi lượt. KHÔNG dùng tiếng Trung hay tiếng Anh."
)

# TTS config
TTS_TARGET_SR = 24000
TTS_PROMPT_TRANSCRIPT = (
    "Toi la chu so huu giong noi nay, va toi dong y cho Google "
    "su dung giong noi nay de tao mo hinh giong noi tong hop."
)

# Prompt audio paths (checked in order)
PROMPT_AUDIO_PATHS = [
    "prompts/consent_audio.wav",
    "/runpod-volume/workspace/consent_audio.wav",
    "/runpod-volume/consent_audio.wav",
]

# Input validation
MAX_AUDIO_DURATION_S = 30
MAX_AUDIO_SIZE_BYTES = 500_000  # ~500KB
