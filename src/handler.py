"""RunPod serverless handler — entrypoint for the all-in-one voice pipeline."""
import base64
import time

import runpod

from src.config import logger, MAX_AUDIO_DURATION_S, MAX_AUDIO_SIZE_BYTES
from src.pipeline import process_turn

# Pre-load all models at import time (RunPod cold start)
logger.info("[HANDLER] Pre-loading models...")
_load_start = time.time()

from src.stt.whisper_stt import load_model as load_stt
from src.llm.qwen_llm import load_model as load_llm
from src.tts.spark_tts import load_model as load_tts

load_stt()
load_llm()
load_tts()

logger.info(f"[HANDLER] All models loaded in {time.time() - _load_start:.1f}s")


def handler(job: dict) -> dict:
    """
    RunPod serverless handler.

    Input schema:
        {
            "audio_b64": "<base64 encoded audio>",
            "conversation": [{"role": "user", "content": "..."}, ...],
            "system_prompt": "optional override"
        }

    Output schema (success):
        {
            "status": "ok",
            "request_id": "abc12345",
            "transcript": "user said this",
            "response": "bot replied this",
            "audio_b64": "<base64 WAV>",
            "latency": {"stt": 0.3, "llm": 0.4, "tts": 0.7, "total": 1.4}
        }

    Output schema (error):
        {
            "status": "error",
            "request_id": "abc12345",
            "failed_stage": "stt|llm|tts",
            "error_code": "...",
            "transcript": "...|null",
            "response": "...|null"
        }
    """
    inp = job.get("input", {})

    # Validate audio_b64
    audio_b64 = inp.get("audio_b64")
    if not audio_b64:
        return {"status": "error", "error_code": "MISSING_AUDIO", "failed_stage": "input"}

    # Decode audio
    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return {"status": "error", "error_code": "INVALID_BASE64", "failed_stage": "input"}

    # Size check
    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        return {
            "status": "error",
            "error_code": "AUDIO_TOO_LARGE",
            "failed_stage": "input",
        }

    # Parse conversation history
    conversation = inp.get("conversation", [])
    if not isinstance(conversation, list):
        conversation = []

    system_prompt = inp.get("system_prompt")

    # Run pipeline
    result = process_turn(
        audio_bytes=audio_bytes,
        conversation=conversation,
        system_prompt=system_prompt,
    )

    return result


# Start RunPod serverless worker
runpod.serverless.start({"handler": handler})
