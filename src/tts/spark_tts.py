"""TTS module — Vi-SparkTTS-0.5B inference on GPU.

CRITICAL constraints (from rp_handler.py):
- torch.take monkey-patch MUST be kept (int64 index fix)
- processor.model = model (NOT link_model())
- global_token_ids handling for voice cloning
"""
import os
import time

import torch
import torchaudio
import numpy as np
import soundfile as sf
from transformers import AutoProcessor, AutoModel

from src.config import (
    logger, DEVICE, HF_TOKEN,
    TTS_MODEL, TTS_TARGET_SR, TTS_PROMPT_TRANSCRIPT,
    PROMPT_AUDIO_PATHS,
)
from src.tts.preprocess import preprocess_for_tts

# ============================================================
# torch.take MONKEY-PATCH — MUST KEEP (int64 index fix)
# ============================================================
_original_torch_take = torch.take


def _patched_torch_take(input_tensor, index):
    if index.dtype != torch.int64:
        index = index.to(torch.int64)
    return _original_torch_take(input_tensor, index)


torch.take = _patched_torch_take

# ============================================================
# Module state
# ============================================================
_model = None
_processor = None
_prompt_path = None
_global_tokens = None  # Cached from first inference with prompt


def _find_prompt_audio() -> str | None:
    """Find voice cloning prompt audio file."""
    for path in PROMPT_AUDIO_PATHS:
        if os.path.exists(path):
            try:
                audio_data, sr = sf.read(path)
                if len(audio_data) == 0:
                    continue
                duration = len(audio_data) / sr
                if duration < 1.0:
                    continue
                max_amp = np.max(np.abs(audio_data))
                if max_amp < 0.001:
                    continue
                logger.info(f"[TTS] Prompt audio: {path} ({duration:.1f}s)")
                return path
            except Exception as e:
                logger.warning(f"[TTS] Bad prompt {path}: {e}")
    return None


def load_model():
    """Load Vi-SparkTTS-0.5B model and processor."""
    global _model, _processor, _prompt_path

    if _model is not None:
        return

    logger.info(f"[TTS] Loading {TTS_MODEL}...")
    t0 = time.time()

    _processor = AutoProcessor.from_pretrained(
        TTS_MODEL, trust_remote_code=True, token=HF_TOKEN,
    )
    _model = AutoModel.from_pretrained(
        TTS_MODEL, trust_remote_code=True, token=HF_TOKEN,
        device_map="cuda",
    ).eval()

    # CRITICAL: processor.model = model (NOT link_model)
    _processor.model = _model

    _prompt_path = _find_prompt_audio()

    logger.info(f"[TTS] Loaded in {time.time() - t0:.1f}s")

    # Warmup
    _warmup()


def _warmup():
    """Run a short inference to avoid cold-start penalty."""
    logger.info("[TTS] Warming up...")
    try:
        inputs = _processor(text=". xin chao.", return_tensors="pt")
        inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}
        _ = inputs.pop("global_token_ids_prompt", None)
        with torch.no_grad():
            _ = _model.generate(**inputs, max_new_tokens=50, do_sample=False)
        torch.cuda.empty_cache()
        logger.info("[TTS] Warmup done")
    except Exception as e:
        logger.warning(f"[TTS] Warmup failed (non-critical): {e}")


def _calc_max_new_tokens(text: str, input_tokens: int) -> int:
    """Calculate max_new_tokens based on text length."""
    text_len = len(text.strip())
    base_ratio = 2.5

    if text_len < 50:
        ratio = base_ratio * 1.5
    elif text_len < 200:
        ratio = base_ratio * 1.1
    elif text_len < 500:
        ratio = base_ratio
    else:
        ratio = base_ratio * 0.8

    estimated = max(int(text_len * 0.3 * ratio), int(input_tokens * ratio))
    estimated = int(estimated * 1.2)
    return min(1800, max(estimated, 600))


def synthesize(text: str) -> tuple[np.ndarray, int, float]:
    """
    Synthesize speech from text.

    Args:
        text: Input text (will be preprocessed/normalized).

    Returns:
        (audio_numpy_array, sample_rate, latency_seconds)
    """
    global _global_tokens

    load_model()

    # Preprocess Vietnamese text
    processed_text = preprocess_for_tts(text)
    logger.info(f"[TTS] Synthesizing: '{processed_text[:60]}...'")

    t0 = time.time()

    # Prepare processor inputs
    proc_args = {"text": processed_text, "return_tensors": "pt"}
    if _prompt_path:
        proc_args["prompt_speech_path"] = _prompt_path
        proc_args["prompt_text"] = TTS_PROMPT_TRANSCRIPT

    inputs = _processor(**proc_args)
    inputs = {k: (v.to(DEVICE) if hasattr(v, "to") else v) for k, v in inputs.items()}

    input_tokens = inputs["input_ids"].shape[-1]

    # Global tokens handling (voice cloning consistency)
    if _global_tokens is None:
        _global_tokens = inputs.pop("global_token_ids_prompt", None)
        if _global_tokens is not None:
            _global_tokens = _global_tokens.to(DEVICE)
            logger.info(f"[TTS] Global tokens cached: {_global_tokens.shape}")
    else:
        _ = inputs.pop("global_token_ids_prompt", None)

    max_new = _calc_max_new_tokens(processed_text, input_tokens)

    # Generate
    with torch.no_grad():
        output_ids = _model.generate(
            **inputs,
            max_new_tokens=max_new,
            do_sample=True,
            temperature=0.7,
            top_k=50,
            top_p=0.8,
            repetition_penalty=1.1,
            eos_token_id=_processor.tokenizer.eos_token_id,
            pad_token_id=_processor.tokenizer.pad_token_id,
        )

    # Decode audio
    output_ids = output_ids.to(DEVICE)
    audio_dict = _processor.decode(
        generated_ids=output_ids,
        global_token_ids_prompt=_global_tokens,
        input_ids_len=input_tokens,
    )

    audio = np.asarray(audio_dict["audio"], dtype=np.float32)
    sr_out = int(audio_dict.get("sampling_rate", TTS_TARGET_SR))

    # Resample if needed
    if sr_out != TTS_TARGET_SR:
        wav = torch.from_numpy(audio).unsqueeze(0)
        audio = torchaudio.functional.resample(wav, sr_out, TTS_TARGET_SR).squeeze(0).numpy()

    # Normalize peak
    max_val = float(np.max(np.abs(audio)))
    if max_val > 0.98:
        audio = (audio * (0.95 / max_val)).astype(np.float32)

    latency = time.time() - t0
    duration = len(audio) / TTS_TARGET_SR
    logger.info(f"[TTS] Done: {duration:.2f}s audio ({latency:.2f}s)")

    torch.cuda.empty_cache()

    return audio, TTS_TARGET_SR, latency
