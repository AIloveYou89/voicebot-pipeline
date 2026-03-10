"""
All-in-One Voice Server — STT + LLM + TTS tren 1 GPU
  STT: Faster-Whisper large-v3
  LLM: Qwen2.5-7B-Instruct (local GPU) hoac GPT-4o-mini (API fallback)
  TTS: StyleTTS2-lite-vi
  Hot-reload: POST /reload to reload handlers.py without restarting models

This file: model loading + Flask routes (NOT reloadable)
handlers.py: all processing logic (reloadable via POST /reload)
"""
import os
import sys
import time
import base64
import json
import struct
import importlib
import threading

import torch
import numpy as np
from flask import Flask, request, jsonify
from flask_cors import CORS

# ============================================================
# Config (model-loading related — NOT reloadable)
# ============================================================

PORT = int(os.environ.get("PORT", "7860"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# STT
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")

# LLM
LLM_MODE = os.environ.get("LLM_MODE", "local")
LLM_MODEL = os.environ.get("LLM_MODEL", "Qwen/Qwen2.5-7B-Instruct")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")

# TTS
STYLETTS2_MODEL_DIR = os.environ.get("STYLETTS2_MODEL_DIR", "/workspace/styletts2_lite_vi")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
STYLETTS2_ALPHA = float(os.environ.get("STYLETTS2_ALPHA", "0.3"))
STYLETTS2_BETA = float(os.environ.get("STYLETTS2_BETA", "0.7"))
STYLETTS2_DIFFUSION_STEPS = int(os.environ.get("STYLETTS2_STEPS", "5"))
STYLETTS2_REF_AUDIO = os.environ.get("STYLETTS2_REF_AUDIO", "")

# Noise suppression
ENABLE_DEEPFILTER = os.environ.get("ENABLE_DEEPFILTER", "0") == "1"

# ============================================================
# Load all models (one-time, stays in GPU memory)
# ============================================================

print(f"[INIT] Device: {DEVICE}")
print(f"[INIT] LLM mode: {LLM_MODE}")
print(f"[INIT] Loading models...")
total_t0 = time.time()

# --- STT: Faster-Whisper ---
print(f"[STT] Loading faster-whisper {WHISPER_MODEL_SIZE}...")
t0 = time.time()
from faster_whisper import WhisperModel
stt_model = WhisperModel(WHISPER_MODEL_SIZE, device=DEVICE, compute_type="float16" if DEVICE == "cuda" else "int8")
print(f"[STT] Loaded in {time.time()-t0:.1f}s")

# --- LLM ---
llm_model = None
llm_tokenizer = None
llm_client = None

if LLM_MODE == "local":
    print(f"[LLM] Loading {LLM_MODEL} on GPU...")
    t0 = time.time()
    from transformers import AutoModelForCausalLM, AutoTokenizer
    llm_tokenizer = AutoTokenizer.from_pretrained(LLM_MODEL, trust_remote_code=True)
    llm_model = AutoModelForCausalLM.from_pretrained(
        LLM_MODEL,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    print(f"[LLM] Loaded in {time.time()-t0:.1f}s")
else:
    print(f"[LLM] Using {LLM_MODEL} via OpenAI API")
    from openai import OpenAI
    llm_client = OpenAI(api_key=OPENAI_API_KEY)
    if not OPENAI_API_KEY:
        print("[LLM] WARNING: OPENAI_API_KEY not set!")

# --- TTS: StyleTTS2-lite-vi ---
print(f"[TTS] Loading StyleTTS2-lite-vi from {STYLETTS2_MODEL_DIR}...")
t0 = time.time()
styletts2_model = None
_tts_method = None

try:
    from styletts2 import tts as StyleTTS2Module
    import glob as _glob

    _ckpt = os.environ.get("STYLETTS2_CKPT", "")
    _config = os.environ.get("STYLETTS2_CONFIG", "")

    if not _ckpt:
        for _ext in ("*.pth", "*.pt", "*.ckpt", "*.safetensors"):
            _found = sorted(_glob.glob(os.path.join(STYLETTS2_MODEL_DIR, "**/" + _ext), recursive=True))
            if _found:
                _ckpt = _found[0]
                break
    if not _config:
        for _ext in ("*.yml", "*.yaml", "*.json"):
            _found = sorted(_glob.glob(os.path.join(STYLETTS2_MODEL_DIR, "**/" + _ext), recursive=True))
            if _found:
                _config = _found[0]
                break

    print(f"[TTS] Model dir contents:")
    for _f in sorted(_glob.glob(os.path.join(STYLETTS2_MODEL_DIR, "*"))):
        print(f"[TTS]   {_f}")

    if _ckpt and _config:
        print(f"[TTS] pip package | ckpt={_ckpt}")
        print(f"[TTS] pip package | config={_config}")
        styletts2_model = StyleTTS2Module.StyleTTS2(
            model_checkpoint_path=_ckpt,
            config_path=_config,
        )
        _tts_method = "pip"
    else:
        raise FileNotFoundError(f"No model/config found in {STYLETTS2_MODEL_DIR}")
except Exception as e1:
    print(f"[TTS] pip package failed: {e1}")
    try:
        if STYLETTS2_MODEL_DIR not in sys.path:
            sys.path.insert(0, STYLETTS2_MODEL_DIR)
        from inference import StyleTTS2 as LocalStyleTTS2
        styletts2_model = LocalStyleTTS2(config_path=_config, models_path=_ckpt).eval().to(DEVICE)
        _tts_method = "local"
        print(f"[TTS] Using local inference.py from {STYLETTS2_MODEL_DIR}")
    except Exception as e2:
        print(f"[TTS] local inference.py also failed: {e2}")
        import traceback
        traceback.print_exc()
        print("[TTS] WARNING: TTS unavailable — audio responses will be empty")

if styletts2_model:
    print(f"[TTS] Loaded via '{_tts_method}' in {time.time()-t0:.1f}s")

# Auto-detect reference audio
_tts_ref_audio_path = STYLETTS2_REF_AUDIO
if not _tts_ref_audio_path:
    import glob as _glob2
    _ref_candidates = sorted(_glob2.glob(os.path.join(STYLETTS2_MODEL_DIR, "reference_audio", "vn_*.wav")))
    if not _ref_candidates:
        _ref_candidates = sorted(_glob2.glob(os.path.join(STYLETTS2_MODEL_DIR, "reference_audio", "*.wav")))
    if _ref_candidates:
        _tts_ref_audio_path = _ref_candidates[0]
        print(f"[TTS] Reference audio: {_tts_ref_audio_path}")
    else:
        print("[TTS] No reference audio found")

# Precompute styles
_tts_styles = None
if styletts2_model and _tts_method == "local" and _tts_ref_audio_path:
    try:
        _speakers = {"id_1": {"path": _tts_ref_audio_path, "lang": "vi", "speed": 1.0}}
        with torch.no_grad():
            _tts_styles = styletts2_model.get_styles(_speakers, denoise=0.6, avg_style=True)
        print(f"[TTS] Precomputed styles from {_tts_ref_audio_path}")
    except Exception as e:
        print(f"[TTS] Failed to precompute styles: {e}")

# --- Noise Suppression ---
df_model = None
df_state = None
df_enhance_fn = None

if ENABLE_DEEPFILTER:
    print("[NS] Loading DeepFilterNet...")
    t0 = time.time()
    try:
        from df.enhance import enhance as _df_enhance, init_df
        df_model, df_state, _ = init_df()
        df_enhance_fn = _df_enhance
        print(f"[NS] DeepFilterNet loaded in {time.time()-t0:.1f}s (sr={df_state.sr()})")
    except Exception as e:
        print(f"[NS] DeepFilterNet not available: {e}")
        ENABLE_DEEPFILTER = False

print(f"[INIT] All models loaded in {time.time()-total_t0:.1f}s")

# ============================================================
# Model context dict — passed to handlers
# ============================================================

_model_ctx = {
    "stt_model": stt_model,
    "llm_model": llm_model,
    "llm_tokenizer": llm_tokenizer,
    "llm_client": llm_client,
    "llm_mode": LLM_MODE,
    "llm_model_name": LLM_MODEL,
    "styletts2_model": styletts2_model,
    "tts_method": _tts_method,
    "tts_styles": _tts_styles,
    "tts_ref_audio_path": _tts_ref_audio_path,
    "tts_sample_rate": TTS_SAMPLE_RATE,
    "df_model": df_model,
    "df_state": df_state,
    "df_enhance_fn": df_enhance_fn,
    "enable_deepfilter": ENABLE_DEEPFILTER,
    "device": DEVICE,
}

# ============================================================
# Initialize handlers
# ============================================================

import handlers
handlers.init(_model_ctx)

# ============================================================
# Warmup
# ============================================================

print("[INIT] Warming up...")
with torch.inference_mode():
    segments, _ = stt_model.transcribe(np.zeros(16000, dtype=np.float32), language="vi")
    list(segments)

    if styletts2_model is not None:
        try:
            if _tts_method == "pip":
                styletts2_model.inference("xin chào",
                    alpha=STYLETTS2_ALPHA, beta=STYLETTS2_BETA,
                    diffusion_steps=STYLETTS2_DIFFUSION_STEPS, embedding_scale=1)
            elif _tts_styles:
                styletts2_model.generate("xin chào", _tts_styles)
            print(f"[TTS] Warmup OK")
        except Exception as e:
            print(f"[TTS] Warmup failed: {e}")

    if llm_model:
        _warmup_text = llm_tokenizer.apply_chat_template(
            [{"role": "system", "content": handlers.SYSTEM_PROMPT},
             {"role": "user", "content": "xin chao"}],
            tokenize=False, add_generation_prompt=True,
        )
        _warmup_ids = llm_tokenizer(_warmup_text, return_tensors="pt").to(DEVICE)
        with torch.no_grad():
            llm_model.generate(**_warmup_ids, max_new_tokens=5)
print("[INIT] Ready!")


# ============================================================
# Flask server
# ============================================================

app = Flask(__name__)
CORS(app)


def _reload_handlers():
    """Reload handlers.py, preserve mute state."""
    old_muted = getattr(handlers, '_muted', False)
    importlib.reload(handlers)
    handlers._muted = old_muted
    handlers.init(_model_ctx)
    return True


@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "device": str(DEVICE),
        "llm_mode": LLM_MODE,
        "noise_suppression": ENABLE_DEEPFILTER,
        "models": {
            "stt": WHISPER_MODEL_SIZE,
            "llm": LLM_MODEL,
            "tts": {
                "engine": "StyleTTS2-lite-vi",
                "method": _tts_method,
                "model_dir": STYLETTS2_MODEL_DIR,
                "sample_rate": TTS_SAMPLE_RATE,
                "diffusion_steps": STYLETTS2_DIFFUSION_STEPS,
            },
        }
    })


@app.route("/reload", methods=["POST"])
def reload_code():
    """Hot-reload handlers.py without restarting models."""
    try:
        _reload_handlers()
        print("[RELOAD] handlers.py reloaded successfully")
        return jsonify({"status": "ok", "message": "handlers.py reloaded"})
    except Exception as e:
        print(f"[RELOAD] Failed: {e}")
        import traceback
        traceback.print_exc()
        return jsonify({"status": "error", "message": str(e)}), 500


@app.route("/chat", methods=["POST"])
def chat():
    data = request.json or {}
    audio_b64 = data.get("audio_b64", "")
    conversation = data.get("conversation", [])

    if not audio_b64:
        return jsonify({"error": "Missing audio_b64"}), 400

    if conversation and isinstance(conversation, list):
        conversation = conversation[-20:]
    else:
        conversation = []

    result, status_code = handlers.handle_chat(audio_b64, conversation)
    return jsonify(result), status_code


@app.route("/tts/b64", methods=["POST"])
def tts_only():
    data = request.json or {}
    text = data.get("text", "").strip()
    if not text:
        return jsonify({"error": "Missing text"}), 400
    t0 = time.time()
    waveform = handlers.tts_synthesize(text)
    tts_time = time.time() - t0
    if waveform is None:
        return jsonify({"error": "TTS failed"}), 500
    return jsonify({
        "audio_b64": base64.b64encode(handlers.waveform_to_wav_bytes(waveform)).decode(),
        "sample_rate": TTS_SAMPLE_RATE,
        "tts_time": round(tts_time, 3),
    })


@app.route("/stt", methods=["POST"])
def stt_only():
    data = request.json or {}
    audio_b64 = data.get("audio_b64", "")
    if not audio_b64:
        return jsonify({"error": "Missing audio_b64"}), 400
    audio_bytes = base64.b64decode(audio_b64)
    t0 = time.time()
    transcript, metrics = handlers.stt_transcribe(audio_bytes)
    return jsonify({
        "transcript": transcript,
        "stt_time": round(time.time() - t0, 3),
        "metrics": metrics,
    })


# ============================================================
# WebSocket endpoint
# ============================================================

try:
    from flask_sock import Sock
    sock = Sock(app)

    @sock.route("/ws")
    def ws_chat(ws):
        print("[WS] Client connected")
        audio_data = None
        cancel_event = threading.Event()
        process_thread = None

        while True:
            try:
                data = ws.receive(timeout=300)
            except Exception:
                break

            if data is None:
                break

            if isinstance(data, bytes):
                audio_data = data
            elif isinstance(data, str):
                try:
                    msg = json.loads(data)
                except json.JSONDecodeError:
                    continue

                if msg.get("type") == "interrupt":
                    cancel_event.set()
                    ws.send(json.dumps({"type": "interrupted"}))
                    print("[WS] Barge-in: client requested interrupt")

                elif msg.get("type") == "process" and audio_data is not None:
                    cancel_event.set()
                    if process_thread and process_thread.is_alive():
                        process_thread.join(timeout=3)

                    client_turn_id = int(msg.get("turn_id", 0))
                    conversation = msg.get("conversation", [])
                    if isinstance(conversation, list):
                        conversation = [
                            m for m in conversation[-20:]
                            if isinstance(m, dict) and m.get("role") in ("user", "assistant")
                        ]
                    else:
                        conversation = []

                    cancel_event = threading.Event()
                    process_thread = threading.Thread(
                        target=handlers.handle_ws_process,
                        args=(ws, audio_data, conversation, cancel_event, client_turn_id),
                        daemon=True,
                    )
                    process_thread.start()
                    audio_data = None

        print("[WS] Client disconnected")

    print("[WS] WebSocket endpoint enabled at /ws")
except ImportError:
    print("[WS] flask-sock not installed — WebSocket disabled")


if __name__ == "__main__":
    print(f"[SERVER] Starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
