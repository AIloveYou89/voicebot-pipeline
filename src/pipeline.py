"""Pipeline orchestrator: STT -> LLM -> TTS.

Phase 1: process_turn() — HTTP sync, full response
Phase 2: process_turn_streaming() — generator, yield audio chunks per phrase
"""
import io
import re
import base64
import time
from uuid import uuid4
from typing import Generator

import numpy as np
import soundfile as sf

from src.config import logger
from src.stt.whisper_stt import transcribe
from src.llm.qwen_llm import generate, generate_stream
from src.tts.spark_tts import synthesize


# Phrase splitting patterns
_SENTENCE_END = re.compile(r'[.?!。？！…]\s*')
_PHRASE_BREAK = re.compile(r'[,;:，；]\s*')
_MAX_PHRASE_WORDS = 20


def _audio_to_b64(audio_np: np.ndarray, sample_rate: int) -> str:
    """Encode numpy audio array to base64 WAV string."""
    buf = io.BytesIO()
    sf.write(buf, audio_np, sample_rate, format="WAV")
    return base64.b64encode(buf.getvalue()).decode("utf-8")


def _split_phrases(buffer: str) -> tuple[list[str], str]:
    """
    Extract complete phrases from token buffer.
    Returns (phrases_ready, remaining_buffer).
    """
    phrases = []
    while True:
        # Sentence boundary first (. ? ! ...)
        m = _SENTENCE_END.search(buffer)
        if m:
            phrase = buffer[:m.end()].strip()
            buffer = buffer[m.end():]
            if phrase:
                phrases.append(phrase)
            continue

        # Phrase boundary (, ; :) — only if long enough
        m = _PHRASE_BREAK.search(buffer)
        if m:
            phrase = buffer[:m.end()].strip()
            rest = buffer[m.end():]
            if phrase and len(phrase.split()) >= 4:
                phrases.append(phrase)
                buffer = rest
                continue
            else:
                break

        # Word count split
        words = buffer.split()
        if len(words) >= _MAX_PHRASE_WORDS:
            phrase = " ".join(words[:_MAX_PHRASE_WORDS])
            buffer = " ".join(words[_MAX_PHRASE_WORDS:])
            phrases.append(phrase)
            continue

        break

    return phrases, buffer


# ============================================================
# Phase 1: HTTP sync (unchanged)
# ============================================================

def process_turn(
    audio_bytes: bytes,
    conversation: list[dict],
    system_prompt: str | None = None,
) -> dict:
    """
    Process one voice conversation turn (Phase 1 — full sync).
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
                "request_id": request_id, "status": "error",
                "failed_stage": "stt", "error_code": "EMPTY_TRANSCRIPT",
                "transcript": None, "response": None,
            }
    except Exception as e:
        logger.error(f"[PIPELINE] STT failed: {e}")
        return {
            "request_id": request_id, "status": "error",
            "failed_stage": "stt", "error_code": "STT_EXCEPTION",
            "transcript": None, "response": None,
        }

    # Stage 2: LLM
    try:
        messages = conversation + [{"role": "user", "content": transcript}]
        response_text, llm_time = generate(messages, system_prompt=system_prompt)
        timings["llm"] = round(llm_time, 3)
        if not response_text:
            return {
                "request_id": request_id, "status": "error",
                "failed_stage": "llm", "error_code": "EMPTY_RESPONSE",
                "transcript": transcript, "response": None,
            }
    except Exception as e:
        logger.error(f"[PIPELINE] LLM failed: {e}")
        return {
            "request_id": request_id, "status": "error",
            "failed_stage": "llm", "error_code": "LLM_EXCEPTION",
            "transcript": transcript, "response": None,
        }

    # Stage 3: TTS
    try:
        audio_np, sample_rate, tts_time = synthesize(response_text)
        timings["tts"] = round(tts_time, 3)
    except Exception as e:
        logger.error(f"[PIPELINE] TTS failed: {e}")
        return {
            "request_id": request_id, "status": "error",
            "failed_stage": "tts", "error_code": "TTS_EXCEPTION",
            "transcript": transcript, "response": response_text,
        }

    audio_b64 = _audio_to_b64(audio_np, sample_rate)
    timings["total"] = round(time.time() - total_start, 3)

    logger.info(
        f"[PIPELINE] {request_id} done: "
        f"STT={timings['stt']}s LLM={timings['llm']}s TTS={timings['tts']}s "
        f"Total={timings['total']}s"
    )

    return {
        "request_id": request_id, "status": "ok",
        "transcript": transcript, "response": response_text,
        "audio_b64": audio_b64, "latency": timings,
    }


# ============================================================
# Phase 2: Streaming (yield audio chunks per phrase)
# ============================================================

def process_turn_streaming(
    audio_bytes: bytes,
    conversation: list[dict],
    system_prompt: str | None = None,
) -> Generator[dict, None, None]:
    """
    Process one voice turn with streaming (Phase 2).

    Yields chunks as they become ready:
    1. {"type": "transcript", "text": "...", "stt_time": 0.4}
    2. {"type": "audio_chunk", "phrase_num": 1, "text": "...", "audio_b64": "...", "tts_time": 0.7}
    3. {"type": "audio_chunk", "phrase_num": 2, ...}
    4. {"type": "done", "response": "full text", "latency": {...}}
    """
    request_id = uuid4().hex[:8]
    timings = {}
    total_start = time.time()

    # Stage 1: STT (blocking — must finish before LLM starts)
    try:
        transcript, stt_time = transcribe(audio_bytes)
        timings["stt"] = round(stt_time, 3)

        if not transcript:
            yield {
                "type": "error", "request_id": request_id,
                "failed_stage": "stt", "error_code": "EMPTY_TRANSCRIPT",
            }
            return
    except Exception as e:
        logger.error(f"[STREAM] STT failed: {e}")
        yield {
            "type": "error", "request_id": request_id,
            "failed_stage": "stt", "error_code": "STT_EXCEPTION",
        }
        return

    yield {
        "type": "transcript",
        "request_id": request_id,
        "text": transcript,
        "stt_time": timings["stt"],
    }

    # Stage 2+3: LLM streaming → phrase detection → TTS per phrase
    messages = conversation + [{"role": "user", "content": transcript}]
    buffer = ""
    full_response = ""
    phrase_num = 0
    llm_start = time.time()
    tts_total = 0.0

    try:
        for token in generate_stream(messages, system_prompt=system_prompt):
            buffer += token
            full_response += token

            # Check for complete phrases
            phrases, buffer = _split_phrases(buffer)

            for phrase in phrases:
                if not phrase.strip():
                    continue

                # First phrase: record LLM time-to-first-phrase
                if phrase_num == 0:
                    timings["llm_first_phrase"] = round(time.time() - llm_start, 3)

                phrase_num += 1
                logger.info(f"[STREAM] Phrase {phrase_num}: '{phrase[:50]}...'")

                # TTS for this phrase (GPU switches from LLM to TTS)
                try:
                    audio_np, sr, tts_time = synthesize(phrase)
                    tts_total += tts_time

                    yield {
                        "type": "audio_chunk",
                        "request_id": request_id,
                        "phrase_num": phrase_num,
                        "text": phrase,
                        "audio_b64": _audio_to_b64(audio_np, sr),
                        "tts_time": round(tts_time, 3),
                    }
                except Exception as e:
                    logger.error(f"[STREAM] TTS failed phrase {phrase_num}: {e}")

    except Exception as e:
        logger.error(f"[STREAM] LLM streaming failed: {e}")
        yield {
            "type": "error", "request_id": request_id,
            "failed_stage": "llm", "error_code": "LLM_STREAM_EXCEPTION",
        }
        return

    # Flush remaining buffer (skip if too short — not worth a TTS call)
    remaining = buffer.strip()
    if remaining and len(remaining.split()) >= 2:
        phrase_num += 1
        if phrase_num == 1:
            timings["llm_first_phrase"] = round(time.time() - llm_start, 3)

        logger.info(f"[STREAM] Flush phrase {phrase_num}: '{remaining[:50]}...'")
        try:
            audio_np, sr, tts_time = synthesize(remaining)
            tts_total += tts_time

            yield {
                "type": "audio_chunk",
                "request_id": request_id,
                "phrase_num": phrase_num,
                "text": remaining,
                "audio_b64": _audio_to_b64(audio_np, sr),
                "tts_time": round(tts_time, 3),
            }
        except Exception as e:
            logger.error(f"[STREAM] TTS failed final phrase: {e}")
    elif remaining:
        logger.info(f"[STREAM] Skipping tiny flush: '{remaining}' (< 2 words)")
        full_response = full_response  # keep in full_response for done message

    timings["tts_total"] = round(tts_total, 3)
    timings["total"] = round(time.time() - total_start, 3)
    timings["phrases"] = phrase_num

    logger.info(
        f"[STREAM] {request_id} done: "
        f"STT={timings['stt']}s LLM_1st={timings.get('llm_first_phrase','?')}s "
        f"TTS_total={timings['tts_total']}s Total={timings['total']}s "
        f"Phrases={phrase_num}"
    )

    yield {
        "type": "done",
        "request_id": request_id,
        "response": full_response.strip(),
        "latency": timings,
    }
