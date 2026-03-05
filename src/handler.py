"""RunPod serverless handler — entrypoint for the all-in-one voice pipeline.

Supports two modes:
- Phase 1 (default): HTTP sync — returns full response at once
- Phase 2 (streaming): Generator — yields audio chunks per phrase

Client chooses mode via input.stream = true/false
"""
import base64
import time

import runpod

from src.config import logger, MAX_AUDIO_DURATION_S, MAX_AUDIO_SIZE_BYTES
from src.pipeline import process_turn, process_turn_streaming

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


def _validate_input(job: dict) -> tuple[bytes | None, list, str | None, bool, dict | None]:
    """Validate and parse job input. Returns (audio_bytes, conversation, system_prompt, stream, error)."""
    inp = job.get("input", {})

    audio_b64 = inp.get("audio_b64")
    if not audio_b64:
        return None, [], None, False, {"status": "error", "error_code": "MISSING_AUDIO", "failed_stage": "input"}

    try:
        audio_bytes = base64.b64decode(audio_b64)
    except Exception:
        return None, [], None, False, {"status": "error", "error_code": "INVALID_BASE64", "failed_stage": "input"}

    if len(audio_bytes) > MAX_AUDIO_SIZE_BYTES:
        return None, [], None, False, {"status": "error", "error_code": "AUDIO_TOO_LARGE", "failed_stage": "input"}

    conversation = inp.get("conversation", [])
    if not isinstance(conversation, list):
        conversation = []

    system_prompt = inp.get("system_prompt")
    stream = inp.get("stream", False)

    return audio_bytes, conversation, system_prompt, stream, None


def handler(job: dict):
    """
    RunPod serverless handler.

    Input:
        {
            "audio_b64": "<base64 audio>",
            "conversation": [...],
            "system_prompt": "optional",
            "stream": false  (default: Phase 1 sync)
                      true   (Phase 2: yield audio chunks)
        }
    """
    audio_bytes, conversation, system_prompt, stream, error = _validate_input(job)
    if error:
        return error

    if stream:
        # Phase 2: streaming — yield chunks as phrases complete
        for chunk in process_turn_streaming(
            audio_bytes=audio_bytes,
            conversation=conversation,
            system_prompt=system_prompt,
        ):
            yield chunk
    else:
        # Phase 1: sync — return full response at once
        result = process_turn(
            audio_bytes=audio_bytes,
            conversation=conversation,
            system_prompt=system_prompt,
        )
        return result


# Start RunPod serverless worker with streaming support
runpod.serverless.start({
    "handler": handler,
    "return_aggregate_stream": True,
})
