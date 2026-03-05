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
    result = _pipe(
        {"raw": audio_np, "sampling_rate": STT_SAMPLE_RATE},
        return_timestamps=False,
    )
    latency = time.time() - t0

    text = result.get("text", "").strip()
    logger.info(f"[STT] '{text[:80]}...' ({latency:.2f}s)")
    return text, latency
