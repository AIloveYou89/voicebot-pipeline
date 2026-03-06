"""STT module — Whisper large-v3 inference on GPU."""
import time
import io

import torch
import torchaudio
import numpy as np
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline

from src.config import (
    logger, DEVICE, HF_TOKEN,
    WHISPER_MODEL, STT_SAMPLE_RATE, STT_LANGUAGE,
)

_pipe = None


def load_model():
    """Load Whisper large-v3 and create ASR pipeline."""
    global _pipe
    if _pipe is not None:
        return

    logger.info(f"[STT] Loading {WHISPER_MODEL}...")
    t0 = time.time()

    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        WHISPER_MODEL,
        torch_dtype=torch.float16,
        low_cpu_mem_usage=True,
        token=HF_TOKEN,
    ).to(DEVICE)

    processor = AutoProcessor.from_pretrained(
        WHISPER_MODEL,
        token=HF_TOKEN,
    )

    _pipe = pipeline(
        "automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        torch_dtype=torch.float16,
        device=DEVICE,
        generate_kwargs={"language": STT_LANGUAGE},
    )

    logger.info(f"[STT] Loaded in {time.time() - t0:.1f}s")


def _ensure_16khz(audio_bytes: bytes) -> np.ndarray:
    """Load audio bytes and resample to 16kHz mono float32."""
    buf = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(buf)

    # Convert to mono
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)

    # Resample if needed
    if sr != STT_SAMPLE_RATE:
        resampler = torchaudio.transforms.Resample(sr, STT_SAMPLE_RATE)
        waveform = resampler(waveform)

    return waveform.squeeze().numpy()


# Known Whisper hallucination patterns (Vietnamese)
_HALLUCINATION_PATTERNS = [
    "subscribe", "kênh", "theo dõi", "đăng ký", "đăng kí",
    "bỏ lỡ", "video hấp dẫn", "hẹn gặp lại", "lalaschool",
    "ghiền mì gõ", "like và share", "bấm chuông",
    "cảm ơn các bạn đã theo dõi", "mọi người", "em là",
]


def _is_hallucination(text: str) -> bool:
    """Detect Whisper hallucination — fake YouTube-style phrases from noise."""
    text_lower = text.lower()
    match_count = sum(1 for p in _HALLUCINATION_PATTERNS if p in text_lower)
    # If 3+ hallucination phrases found, very likely fake
    return match_count >= 3


def transcribe(audio_bytes: bytes) -> tuple[str, float]:
    """
    Transcribe audio bytes to Vietnamese text.

    Args:
        audio_bytes: Raw audio file bytes (WAV, MP3, etc.)

    Returns:
        (transcript_text, latency_seconds)
    """
    load_model()

    t0 = time.time()
    audio_np = _ensure_16khz(audio_bytes)

    # Check if audio is mostly silence (very low threshold — only pure silence)
    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    logger.info(f"[STT] Audio RMS={rms:.4f}, duration={len(audio_np)/STT_SAMPLE_RATE:.1f}s")
    if rms < 0.001:
        latency = time.time() - t0
        logger.info(f"[STT] Pure silence (RMS={rms:.4f}), skipping ({latency:.2f}s)")
        return "", latency

    result = _pipe(
        {"raw": audio_np, "sampling_rate": STT_SAMPLE_RATE},
        return_timestamps=False,
    )
    latency = time.time() - t0

    text = result.get("text", "").strip()

    # Filter Whisper hallucinations (only if 3+ patterns match — very confident)
    if _is_hallucination(text):
        logger.warning(f"[STT] Hallucination filtered: '{text[:80]}' ({latency:.2f}s)")
        return "", latency

    logger.info(f"[STT] '{text[:80]}...' ({latency:.2f}s)")
    return text, latency
