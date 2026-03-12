"""
All-in-One Voice Server — STT + LLM + TTS
  STT: Faster-Whisper large-v3 (GPU)
  LLM: Groq API (default) hoặc local Qwen (fallback)
  TTS: F5-TTS-Vietnamese-ViVoice (GPU)
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

PORT = int(os.environ.get("PORT", "5300"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# STT
WHISPER_MODEL_SIZE = os.environ.get("WHISPER_MODEL", "large-v3")

# LLM
LLM_MODE = os.environ.get("LLM_MODE", "groq")  # "groq", "local", "api"
LLM_MODEL = os.environ.get("LLM_MODEL", "")
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "")
GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")

# TTS (F5-TTS)
F5_TTS_REPO_DIR = os.environ.get("F5_TTS_REPO_DIR", "/workspace/F5-TTS-Vietnamese")
F5_TTS_CKPT = os.environ.get("F5_TTS_CKPT", "/workspace/F5-TTS-Vietnamese-ViVoice/model_last.pt")
F5_TTS_VOCAB = os.environ.get("F5_TTS_VOCAB", "/workspace/F5-TTS-Vietnamese/vocab.txt")
F5_TTS_REF_AUDIO = os.environ.get("F5_TTS_REF_AUDIO", "/workspace/F5-TTS-Vietnamese/ref.wav")
F5_TTS_REF_TEXT = os.environ.get("F5_TTS_REF_TEXT", "cả hai bên hãy cố gắng hiểu cho nhau")
TTS_SAMPLE_RATE = int(os.environ.get("TTS_SAMPLE_RATE", "24000"))
F5_TTS_SPEED = float(os.environ.get("F5_TTS_SPEED", "1.0"))

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

if LLM_MODE == "groq":
    LLM_MODEL = GROQ_MODEL
    if not GROQ_API_KEY:
        print("[LLM] FATAL: GROQ_API_KEY not set! Get one at https://console.groq.com/keys")
        sys.exit(1)
    print(f"[LLM] Using {GROQ_MODEL} via Groq API (GPU free for TTS!)")
    from openai import OpenAI as _OpenAI
    llm_client = _OpenAI(
        api_key=GROQ_API_KEY,
        base_url="https://api.groq.com/openai/v1",
    )
elif LLM_MODE == "local":
    if not LLM_MODEL:
        LLM_MODEL = "Qwen/Qwen2.5-3B-Instruct"
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
    if not LLM_MODEL:
        LLM_MODEL = "gpt-4o-mini"
    print(f"[LLM] Using {LLM_MODEL} via OpenAI API")
    from openai import OpenAI
    llm_client = OpenAI(api_key=OPENAI_API_KEY)
    if not OPENAI_API_KEY:
        print("[LLM] WARNING: OPENAI_API_KEY not set!")

# --- TTS: F5-TTS-Vietnamese ---
print(f"[TTS] Loading F5-TTS-Vietnamese...")
print(f"[TTS]   ckpt:  {F5_TTS_CKPT}")
print(f"[TTS]   vocab: {F5_TTS_VOCAB}")
print(f"[TTS]   ref:   {F5_TTS_REF_AUDIO}")
t0 = time.time()
f5_tts_model = None

try:
    from f5_tts.api import F5TTS
    f5_tts_model = F5TTS(
        model="F5TTS_Base",
        ckpt_file=F5_TTS_CKPT,
        vocab_file=F5_TTS_VOCAB,
        device=DEVICE,
    )
    print(f"[TTS] F5-TTS loaded in {time.time()-t0:.1f}s")
except Exception as e:
    print(f"[TTS] F5-TTS load failed: {e}")
    import traceback
    traceback.print_exc()
    print("[TTS] WARNING: TTS unavailable — audio responses will be empty")

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
    "f5_tts_model": f5_tts_model,
    "f5_tts_ref_audio": F5_TTS_REF_AUDIO,
    "f5_tts_ref_text": F5_TTS_REF_TEXT,
    "f5_tts_speed": F5_TTS_SPEED,
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

    if f5_tts_model is not None:
        try:
            handlers.tts_synthesize("xin chào")
            print(f"[TTS] Warmup OK")
            # Cache reference audio to avoid reloading from disk on every TTS call
            try:
                import torchaudio
                ref_wav, ref_sr = torchaudio.load(F5_TTS_REF_AUDIO)
                _model_ctx["f5_tts_ref_audio_cached"] = (ref_wav, ref_sr)
                print(f"[TTS] Reference audio cached: {F5_TTS_REF_AUDIO} ({ref_sr}Hz)")
            except Exception as cache_e:
                print(f"[TTS] Ref audio cache skipped: {cache_e}")
        except Exception as e:
            print(f"[TTS] Warmup failed: {e}")

    # Pre-synthesize backchannel clips
    if f5_tts_model is not None:
        try:
            handlers.pre_synthesize_backchannels()
        except Exception as e:
            print(f"[BC] Backchannel pre-synthesis failed: {e}")

    if llm_model and llm_tokenizer:
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
                "engine": "F5-TTS-Vietnamese-ViVoice",
                "ckpt": F5_TTS_CKPT,
                "vocab": F5_TTS_VOCAB,
                "ref_audio": F5_TTS_REF_AUDIO,
                "sample_rate": TTS_SAMPLE_RATE,
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
    nfe_step = data.get("nfe_step", None)
    t0 = time.time()
    waveform = handlers.tts_synthesize(text, nfe_step=nfe_step)
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
# Metrics dashboard
# ============================================================

@app.route("/metrics/dashboard", methods=["GET"])
def metrics_dashboard():
    """Quality metrics dashboard — tổng hợp STT/LLM/TTS quality."""
    try:
        from quality_metrics import summarize_metrics
        last_n = request.args.get("last", 100, type=int)
        metrics_file = os.environ.get("METRICS_FILE", "/workspace/voicebot-pipeline/call_metrics.jsonl")
        summary = summarize_metrics(metrics_file, last_n=last_n)
        return jsonify(summary)
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/metrics/wer", methods=["POST"])
def metrics_wer():
    """Tính WER giữa ground truth và STT output.
    POST {"reference": "tôi muốn đặt lịch", "hypothesis": "tôi muốn đặt lệch"}
    """
    try:
        from quality_metrics import compute_wer
        data = request.json or {}
        ref = data.get("reference", "")
        hyp = data.get("hypothesis", "")
        if not ref:
            return jsonify({"error": "Missing reference"}), 400
        wer = compute_wer(ref, hyp)
        grade = "good" if wer < 0.05 else "ok" if wer < 0.1 else "poor" if wer < 0.15 else "bad"
        return jsonify({"wer": wer, "grade": grade, "reference": ref, "hypothesis": hyp})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


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

        handlers._cleanup_session(ws)
        print("[WS] Client disconnected")

    print("[WS] WebSocket endpoint enabled at /ws")
except ImportError:
    print("[WS] flask-sock not installed — WebSocket disabled")


if __name__ == "__main__":
    print(f"[SERVER] Starting on port {PORT}")
    app.run(host="0.0.0.0", port=PORT, debug=False)
