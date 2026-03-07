"""
Matcha-TTS Server — Vietnamese TTS (giọng Nguyễn Ngọc Ngạn)

Non-autoregressive TTS, nhanh hơn SparkTTS ~2-3x

Usage:
    pip install -r requirements-matcha.txt
    python matcha_tts_server.py

API:
    POST /tts/b64  {"text": "Xin chào"}  → {"audio_b64": "...", "tts_time": 0.5}
    GET  /health
"""
import os
import io
import time
import base64
import wave

import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS
from huggingface_hub import hf_hub_download

from matcha.cli import get_torch_device, load_matcha, load_vocoder, process_text, to_waveform

# ============================================================
# Config
# ============================================================

PORT = int(os.environ.get("PORT", "7860"))
MODEL_REPO = "doof-ferb/matcha_ngngngan"
N_TIMESTEPS = int(os.environ.get("MATCHA_STEPS", "10"))  # 10 steps = fast, 50 = high quality
TEMPERATURE = float(os.environ.get("MATCHA_TEMP", "0.667"))
LENGTH_SCALE = float(os.environ.get("MATCHA_LENGTH_SCALE", "0.95"))
SAMPLE_RATE = 22050

# ============================================================
# Load model
# ============================================================

print("[MATCHA] Loading model...")
t0 = time.time()

MODEL_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="ckpt/checkpoint_epoch420_slim.pt")
VOCODER_PATH = hf_hub_download(repo_id=MODEL_REPO, filename="hifigan/g_02500000")

DEVICE = get_torch_device()
MODEL = load_matcha(MODEL_PATH, DEVICE)
VOCODER, DENOISER = load_vocoder(VOCODER_PATH, DEVICE)

print(f"[MATCHA] Loaded in {time.time() - t0:.1f}s | Device: {DEVICE}")

# Warmup
print("[MATCHA] Warming up...")
with torch.inference_mode():
    out = process_text("xin chào", DEVICE)
    mel_out = MODEL.synthesise(out["x"], out["x_lengths"], n_timesteps=N_TIMESTEPS, temperature=TEMPERATURE, spks=None, length_scale=LENGTH_SCALE)
    _ = to_waveform(mel_out["mel"], VOCODER, DENOISER)
print("[MATCHA] Ready!")

# ============================================================
# TTS function
# ============================================================

@torch.inference_mode()
def synthesize(text: str) -> tuple:
    """Synthesize text to audio. Returns (wav_bytes, tts_time)."""
    t0 = time.time()

    # Process text to phonemes
    text_out = process_text(text, DEVICE)

    # Generate mel spectrogram
    mel_out = MODEL.synthesise(
        text_out["x"],
        text_out["x_lengths"],
        n_timesteps=N_TIMESTEPS,
        temperature=TEMPERATURE,
        spks=None,
        length_scale=LENGTH_SCALE,
    )

    # Mel to waveform via HiFiGAN vocoder
    waveform = to_waveform(mel_out["mel"], VOCODER, DENOISER).numpy()

    tts_time = time.time() - t0

    # Convert to WAV bytes
    audio_int16 = (waveform * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    return buf.getvalue(), tts_time


# ============================================================
# Flask server
# ============================================================

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "engine": "matcha-tts",
        "model": MODEL_REPO,
        "device": str(DEVICE),
        "n_timesteps": N_TIMESTEPS,
        "temperature": TEMPERATURE,
    })


@app.route("/tts/b64", methods=["POST"])
def tts_b64():
    data = request.json or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400
    if len(text) > 2000:
        return jsonify({"error": "Text too long (max 2000 chars)"}), 400

    try:
        wav_bytes, tts_time = synthesize(text)
        audio_b64 = base64.b64encode(wav_bytes).decode()
        duration = len(wav_bytes) / (SAMPLE_RATE * 2)  # approximate

        print(f"[TTS] {tts_time:.2f}s | {len(text)} chars | {duration:.1f}s audio")

        return jsonify({
            "audio_b64": audio_b64,
            "sample_rate": SAMPLE_RATE,
            "tts_time": round(tts_time, 3),
        })
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/tts", methods=["POST"])
def tts_wav():
    """Return WAV file directly."""
    data = request.json or {}
    text = data.get("text", "").strip()

    if not text:
        return jsonify({"error": "Missing 'text'"}), 400

    try:
        wav_bytes, tts_time = synthesize(text)
        from flask import Response
        return Response(wav_bytes, mimetype="audio/wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print(f"[MATCHA] Server starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
