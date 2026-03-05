"""Pipeline orchestrator: STT -> LLM -> TTS (Phase 1: HTTP sync)."""
import io
import base64
import time
from uuid import uuid4

import numpy as np
import soundfile as sf

from src.config import logger
from src.stt.whisper_stt import transcribe
from src.llm.qwen_llm import generate
from src.tts.spark_tts import synthesize


def _audio_to_b64(audio_np: np.ndarray, sample_rate: int) -> str:
    """Encode numpy audio array to base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def process_turn(
    audio_bytes: bytes,
    conversation: list[dict],
    system_prompt: str | None = None,
) -> dict:
    """
    Process one voice conversation turn.

    Flow: audio_bytes -> STT -> LLM -> TTS -> audio_b64

    Args:
        audio_bytes: Input audio (WAV/MP3 bytes).
        conversation: Chat history [{"role": "user"|"assistant", "content": "..."}].
        system_prompt: Optional system prompt override.

    Returns:
        dict with status, transcript, response, audio_b64, latency breakdown.
    """
    request_id = uuid4().hex[:8]
    timings = {}
    total_start = time.time()

    # Stage 1: STT
    try:
        transcript, stt_time = transcribe(audio_bytes)
        timings["stt"] = round(stt_time, 3)

        if not transcript:
            return {
                "request_id": request_id,
                "status": "error",
                "failed_stage": "stt",
                "error_code": "EMPTY_TRANSCRIPT",
                "transcript": None,
                "response": None,
            }
    except Exception as e:
        logger.error(f"[PIPELINE] STT failed: {e}")
        return {
            "request_id": request_id,
            "status": "error",
            "failed_stage": "stt",
            "error_code": "STT_EXCEPTION",
            "transcript": None,
            "response": None,
        }

    # Stage 2: LLM
    try:
        messages = conversation + [{"role": "user", "content": transcript}]
        response_text, llm_time = generate(
            messages, system_prompt=system_prompt,
        )
        timings["llm"] = round(llm_time, 3)

        if not response_text:
            return {
                "request_id": request_id,
                "status": "error",
                "failed_stage": "llm",
                "error_code": "EMPTY_RESPONSE",
                "transcript": transcript,
                "response": None,
            }
    except Exception as e:
        logger.error(f"[PIPELINE] LLM failed: {e}")
        return {
            "request_id": request_id,
            "status": "error",
            "failed_stage": "llm",
            "error_code": "LLM_EXCEPTION",
            "transcript": transcript,
            "response": None,
        }

    # Stage 3: TTS
    try:
        audio_np, sample_rate, tts_time = synthesize(response_text)
        timings["tts"] = round(tts_time, 3)
    except Exception as e:
        logger.error(f"[PIPELINE] TTS failed: {e}")
        return {
            "request_id": request_id,
            "status": "error",
            "failed_stage": "tts",
            "error_code": "TTS_EXCEPTION",
            "transcript": transcript,
            "response": response_text,
        }

    # Encode audio
    audio_b64 = _audio_to_b64(audio_np, sample_rate)
    timings["total"] = round(time.time() - total_start, 3)

    logger.info(
        f"[PIPELINE] {request_id} done: "
        f"STT={timings['stt']}s LLM={timings['llm']}s TTS={timings['tts']}s "
        f"Total={timings['total']}s"
    )

    return {
        "request_id": request_id,
        "status": "ok",
        "transcript": transcript,
        "response": response_text,
        "audio_b64": audio_b64,
        "latency": timings,
    }
