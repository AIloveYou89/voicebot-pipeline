"""
All-in-One Voice Server — STT + LLM + TTS trên 1 GPU
  STT: Faster-Whisper large-v3
  LLM: Qwen2.5-7B-Instruct-AWQ
  TTS: Matcha-TTS (giọng Nguyễn Ngọc Ngạn)

Single endpoint: POST /chat  audio_b64 in → audio_b64 out
Chạy trên RunPod pod với GPU.

Usage:
    pip install faster-whisper flask flask-cors huggingface_hub
    pip install git+https://github.com/phineas-pta/MatchaTTS_ngngngan.git
    python all_in_one_server.py
"""
import os
import io
import time
import base64
import wave
import re

import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# Config
# ============================================================

PORT = int(os.environ.get("PORT", "7860"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# STT
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")

# LLM
QWEN_MODEL = os.environ.get("QWEN_MODEL", "Qwen/Qwen2.5-7B-Instruct-AWQ")
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "96"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
SYSTEM_PROMPT = (
    "Bạn là trợ lý AI nói tiếng Việt. "
    "Quy tắc BẮT BUỘC: CHỈ trả lời bằng tiếng Việt thuần túy. "
    "TUYỆT ĐỐI KHÔNG dùng tiếng Trung, tiếng Anh, hay bất kỳ ngôn ngữ nào khác. "
    "Trả lời ngắn gọn, tự nhiên, tối đa 2 câu."
)

# TTS
MATCHA_REPO = "doof-ferb/matcha_ngngngan"
MATCHA_STEPS = int(os.environ.get("MATCHA_STEPS", "10"))
MATCHA_TEMP = float(os.environ.get("MATCHA_TEMP", "0.667"))
MATCHA_LENGTH_SCALE = float(os.environ.get("MATCHA_LENGTH_SCALE", "0.95"))
TTS_SAMPLE_RATE = 22050

# ============================================================
# Load all models
# ============================================================

print(f"[INIT] Device: {DEVICE}")
print(f"[INIT] Loading 3 models...")
total_t0 = time.time()

# --- STT: Faster-Whisper ---
print(f"[STT] Loading faster-whisper {WHISPER_MODEL_SIZE}...")
t0 = time.time()
from faster_whisper import WhisperModel
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")
print(f"[STT] Loaded in {time.time()-t0:.1f}s")

# --- LLM: Qwen ---
print(f"[LLM] Loading {QWEN_MODEL}...")
t0 = time.time()
from transformers import AutoModelForCausalLM, AutoTokenizer
HF_TOKEN = os.environ.get("HF_TOKEN")

llm_tokenizer = AutoTokenizer.from_pretrained(QWEN_MODEL, trust_remote_code=True, token=HF_TOKEN)
llm_model = AutoModelForCausalLM.from_pretrained(
    QWEN_MODEL, torch_dtype=torch.float16, device_map="auto",
    trust_remote_code=True, token=HF_TOKEN,
)
print(f"[LLM] Loaded in {time.time()-t0:.1f}s")

# --- TTS: Matcha-TTS ---
print(f"[TTS] Loading Matcha-TTS...")
t0 = time.time()
from huggingface_hub import hf_hub_download
from matcha.cli import get_torch_device, load_matcha, load_vocoder, process_text, to_waveform

matcha_model_path = hf_hub_download(repo_id=MATCHA_REPO, filename="ckpt/checkpoint_epoch420_slim.pt")
matcha_vocoder_path = hf_hub_download(repo_id=MATCHA_REPO, filename="hifigan/g_02500000")
matcha_model = load_matcha(matcha_model_path, DEVICE)
matcha_vocoder, matcha_denoiser = load_vocoder(matcha_vocoder_path, DEVICE)
print(f"[TTS] Loaded in {time.time()-t0:.1f}s")

print(f"[INIT] All models loaded in {time.time()-total_t0:.1f}s")

# Warmup
print("[INIT] Warming up...")
with torch.inference_mode():
    # STT warmup
    segments, _ = stt_model.transcribe(np.zeros(16000, dtype=np.float32), language="vi")
    list(segments)
    # TTS warmup
    out = process_text("xin chào", DEVICE)
    mel = matcha_model.synthesise(out["x"], out["x_lengths"], n_timesteps=MATCHA_STEPS, temperature=MATCHA_TEMP, spks=None, length_scale=MATCHA_LENGTH_SCALE)
    _ = to_waveform(mel["mel"], matcha_vocoder, matcha_denoiser)
print("[INIT] Ready!")

# ============================================================
# Helper functions
# ============================================================

def stt_transcribe(audio_bytes):
    """Transcribe audio bytes to text."""
    buf = io.BytesIO(audio_bytes)
    import soundfile as sf
    audio_np, sr = sf.read(buf)
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)

    segments, info = stt_model.transcribe(audio_np, language="vi")
    text = " ".join([s.text for s in segments]).strip()
    return text


def llm_generate(user_text, conversation=None):
    """Generate LLM response."""
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation:
        messages.extend(conversation)
    messages.append({"role": "user", "content": user_text})

    text_input = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(text_input, return_tensors="pt").to(llm_model.device)

    with torch.inference_mode():
        outputs = llm_model.generate(
            **inputs,
            max_new_tokens=LLM_MAX_TOKENS,
            temperature=LLM_TEMPERATURE,
            do_sample=True,
            top_p=0.9,
        )

    new_tokens = outputs[0][inputs["input_ids"].shape[1]:]
    response = llm_tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


@torch.inference_mode()
def tts_synthesize(text):
    """Synthesize text to WAV bytes."""
    text = text.strip()
    if not text:
        return None

    text_out = process_text(text, DEVICE)
    mel_out = matcha_model.synthesise(
        text_out["x"], text_out["x_lengths"],
        n_timesteps=MATCHA_STEPS, temperature=MATCHA_TEMP,
        spks=None, length_scale=MATCHA_LENGTH_SCALE,
    )
    waveform = to_waveform(mel_out["mel"], matcha_vocoder, matcha_denoiser).numpy()

    audio_int16 = (waveform * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(TTS_SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ============================================================
# Flask server
# ============================================================

app = Flask(__name__)
CORS(app)


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "models": {
            "stt": WHISPER_MODEL_SIZE,
            "llm": QWEN_MODEL,
            "tts": MATCHA_REPO,
        }
    })


@app.route("/chat", methods=["POST"])
def chat():
    """Full pipeline: audio_b64 in → audio_b64 out."""
    data = request.json or {}
    audio_b64 = data.get("audio_b64", "")
    conversation = data.get("conversation", [])

    if not audio_b64:
        return jsonify({"error": "Missing audio_b64"}), 400

    audio_bytes = base64.b64decode(audio_b64)
    total_start = time.time()

    # STT
    try:
        t0 = time.time()
        transcript = stt_transcribe(audio_bytes)
        stt_time = time.time() - t0
        print(f"[STT] {stt_time:.2f}s: {transcript}")

        if not transcript:
            return jsonify({"error": "Empty transcript"}), 200
    except Exception as e:
        print(f"[STT] Error: {e}")
        return jsonify({"error": f"STT failed: {e}"}), 500

    # LLM
    try:
        t0 = time.time()
        reply = llm_generate(transcript, conversation)
        llm_time = time.time() - t0
        print(f"[LLM] {llm_time:.2f}s: {reply}")
    except Exception as e:
        print(f"[LLM] Error: {e}")
        return jsonify({"error": f"LLM failed: {e}"}), 500

    # TTS
    try:
        t0 = time.time()
        wav_bytes = tts_synthesize(reply)
        tts_time = time.time() - t0
        print(f"[TTS] {tts_time:.2f}s")

        audio_out_b64 = base64.b64encode(wav_bytes).decode() if wav_bytes else ""
    except Exception as e:
        print(f"[TTS] Error: {e}")
        return jsonify({"error": f"TTS failed: {e}"}), 500

    total = time.time() - total_start
    print(f"[TOTAL] {total:.2f}s | STT {stt_time:.2f} + LLM {llm_time:.2f} + TTS {tts_time:.2f}")

    return jsonify({
        "transcript": transcript,
        "response": reply,
        "audio_b64": audio_out_b64,
        "sample_rate": TTS_SAMPLE_RATE,
        "latency": {
            "stt": round(stt_time, 2),
            "llm": round(llm_time, 2),
            "tts": round(tts_time, 2),
            "total": round(total, 2),
        }
    })


@app.route("/tts/b64", methods=["POST"])
def tts_only():
    """TTS only endpoint (for testing)."""
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400

    t0 = time.time()
    wav_bytes = tts_synthesize(text)
    tts_time = time.time() - t0

    if not wav_bytes:
        return jsonify({"error": "TTS failed"}), 500

    return jsonify({
        "audio_b64": base64.b64encode(wav_bytes).decode(),
        "sample_rate": TTS_SAMPLE_RATE,
        "tts_time": round(tts_time, 3),
    })


@app.route("/stt", methods=["POST"])
def stt_only():
    """STT only endpoint (for testing)."""
    data = request.json or {}
    audio_b64 = data.get("audio_b64", "")
    if not audio_b64:
        return jsonify({"error": "Missing audio_b64"}), 400

    audio_bytes = base64.b64decode(audio_b64)
    t0 = time.time()
    transcript = stt_transcribe(audio_bytes)
    stt_time = time.time() - t0

    return jsonify({
        "transcript": transcript,
        "stt_time": round(stt_time, 3),
    })


if __name__ == "__main__":
    print(f"[SERVER] Starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
