"""TTS module — VieNeu-TTS inference on GPU with voice cloning.

Drop-in replacement for spark_tts.py with same interface:
    load_model(), synthesize(text) -> (audio_np, sample_rate, latency)
"""
import os
import time

import numpy as np
import soundfile as sf
import torchaudio
import torch

from src.config import (
    logger, TTS_TARGET_SR, TTS_PROMPT_TRANSCRIPT,
    PROMPT_AUDIO_PATHS,
)
from src.tts.preprocess import normalize_text_vn

_tts = None
_ref_audio_path = None


def _find_ref_audio() -> str | None:
    """Find voice cloning reference audio file."""
    for path in PROMPT_AUDIO_PATHS:
        if os.path.exists(path):
            try:
                audio_data, sr = sf.read(path)
                if len(audio_data) == 0:
                    continue
                duration = len(audio_data) / sr
                if duration < 1.0:
                    continue
                logger.info(f"[TTS] Reference audio found: {path} ({duration:.1f}s)")
                return path
            except Exception as e:
                logger.warning(f"[TTS] Bad ref audio {path}: {e}")
    return None


def load_model():
    """Load VieNeu-TTS model."""
    global _tts, _ref_audio_path
    if _tts is not None:
        return

    logger.info("[TTS] Loading VieNeu-TTS (0.5B)...")
    t0 = time.time()

    from vieneu import Vieneu
    _tts = Vieneu()

    # Find reference audio for voice cloning
    _ref_audio_path = _find_ref_audio()
    if _ref_audio_path:
        logger.info(f"[TTS] Voice cloning enabled with: {_ref_audio_path}")
    else:
        logger.warning("[TTS] No reference audio found — using default voice")

    logger.info(f"[TTS] VieNeu loaded in {time.time() - t0:.1f}s")

    # Warmup
    _warmup()


def _warmup():
    """Run a short inference to warm up the model."""
    logger.info("[TTS] Warming up...")
    try:
        infer_args = {"text": "Xin chào."}
        if _ref_audio_path:
            infer_args["ref_audio"] = _ref_audio_path
            infer_args["ref_text"] = TTS_PROMPT_TRANSCRIPT
        _ = _tts.infer(**infer_args)
        logger.info("[TTS] Warmup done")
    except Exception as e:
        logger.warning(f"[TTS] Warmup failed (non-critical): {e}")


def synthesize(text: str) -> tuple[np.ndarray, int, float]:
    """
    Synthesize speech from text with voice cloning.

    Args:
        text: Input text (will be preprocessed/normalized).

    Returns:
        (audio_numpy_array, sample_rate, latency_seconds)
    """
    load_model()

    # Preprocess Vietnamese text
    processed_text = normalize_text_vn(text)
    if not processed_text:
        processed_text = text.strip()

    logger.info(f"[TTS] Synthesizing: '{processed_text[:60]}...'")
    t0 = time.time()

    # Build inference args — with voice cloning if ref audio available
    infer_args = {"text": processed_text}
    if _ref_audio_path:
        infer_args["ref_audio"] = _ref_audio_path
        infer_args["ref_text"] = TTS_PROMPT_TRANSCRIPT

    result = _tts.infer(**infer_args)

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
