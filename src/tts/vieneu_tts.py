"""TTS module — VieNeu-TTS inference on GPU.

Drop-in replacement for spark_tts.py with same interface:
    load_model(), synthesize(text) -> (audio_np, sample_rate, latency)
"""
import time

import numpy as np
import torchaudio
import torch

from src.config import logger, TTS_TARGET_SR
from src.tts.preprocess import normalize_text_vn

_tts = None


def load_model():
    """Load VieNeu-TTS model."""
    global _tts
    if _tts is not None:
        return

    logger.info("[TTS] Loading VieNeu-TTS (0.5B)...")
    t0 = time.time()

    from vieneu import Vieneu
    _tts = Vieneu()

    logger.info(f"[TTS] VieNeu loaded in {time.time() - t0:.1f}s")

    # Warmup
    _warmup()


def _warmup():
    """Run a short inference to warm up the model."""
    logger.info("[TTS] Warming up...")
    try:
        _ = _tts.infer(text="Xin chào.")
        logger.info("[TTS] Warmup done")
    except Exception as e:
        logger.warning(f"[TTS] Warmup failed (non-critical): {e}")


def synthesize(text: str) -> tuple[np.ndarray, int, float]:
    """
    Synthesize speech from text.

    Args:
        text: Input text (will be preprocessed/normalized).

    Returns:
        (audio_numpy_array, sample_rate, latency_seconds)
    """
    load_model()

    # Preprocess Vietnamese text (reuse existing normalizer, skip SparkTTS-specific leading dot)
    processed_text = normalize_text_vn(text)
    if not processed_text:
        processed_text = text.strip()

    logger.info(f"[TTS] Synthesizing: '{processed_text[:60]}...'")
    t0 = time.time()

    # VieNeu returns audio dict with 'audio' and 'sampling_rate'
    result = _tts.infer(text=processed_text)

    # Extract audio - VieNeu returns numpy array or dict
    if isinstance(result, dict):
        audio = np.asarray(result["audio"], dtype=np.float32)
        sr_out = int(result.get("sampling_rate", 24000))
    else:
        # Direct numpy array
        audio = np.asarray(result, dtype=np.float32)
        sr_out = 24000

    # Resample to target SR if needed
    if sr_out != TTS_TARGET_SR:
        wav = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(wav, sr_out, TTS_TARGET_SR).squeeze(0).numpy()

    # Normalize peak
    max_val = float(np.max(np.abs(audio)))
    if max_val > 0.01:
        if max_val > 0.98:
            audio = (audio * (0.95 / max_val)).astype(np.float32)
    else:
        logger.warning("[TTS] Audio is near-silent")

    latency = time.time() - t0
    duration = len(audio) / TTS_TARGET_SR
    logger.info(f"[TTS] Done: {duration:.2f}s audio ({latency:.2f}s)")

    return audio, TTS_TARGET_SR, latency
