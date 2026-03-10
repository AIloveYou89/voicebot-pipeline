"""
Reloadable handlers for voicebot server.
Hot-reload: POST /reload — models stay in GPU memory, only this file reloads.

Edit THIS file for logic changes (prompts, patterns, TTS params, etc.)
then call POST /reload to apply without restarting.
"""
import os
import io
import time
import base64
import wave
import re
import json
import threading
from math import gcd
from datetime import datetime, timezone

import torch
import numpy as np

# ============================================================
# Model context — injected by server via init()
# ============================================================
_ctx = {}


def init(ctx):
    """Inject model references from the main server. Called on startup + each reload."""
    global _ctx
    _ctx = ctx


# ============================================================
# Behavioral config (re-read from env on each reload)
# ============================================================
LLM_MAX_TOKENS = int(os.environ.get("LLM_MAX_TOKENS", "96"))
LLM_TEMPERATURE = float(os.environ.get("LLM_TEMPERATURE", "0.5"))
SYSTEM_PROMPT = os.environ.get("SYSTEM_PROMPT",
    "Bạn là trợ lý đặt lịch hẹn qua điện thoại cho công ty. "
    "CHỈ nói tiếng Việt. TUYỆT ĐỐI KHÔNG dùng tiếng Trung Quốc, tiếng Anh, hay bất kỳ ngôn ngữ nào khác. "
    "TUYỆT ĐỐI KHÔNG dùng ký tự Trung Quốc (Hán tự). Chỉ dùng chữ Việt có dấu. "
    "KHÔNG BAO GIỜ nói: 'hẹn gặp lại', 'subscribe', 'video', 'kênh', 'like share', 'đăng ký'. "
    "Bạn KHÔNG PHẢI YouTuber, streamer, hay MC. Bạn là nhân viên telesale. "
    "KHÔNG dùng từ 'ạ' ở cuối câu. Nói tự nhiên, không cần từ đệm cuối câu. "
    "Trả lời ngắn gọn 1-2 câu, tự nhiên như người thật. "
    "Nếu khách hỏi gì ngoài phạm vi, nói: 'Dạ, em chưa có thông tin về vấn đề này.'"
)

# TTS behavioral params
STYLETTS2_ALPHA = float(os.environ.get("STYLETTS2_ALPHA", "0.3"))
STYLETTS2_BETA = float(os.environ.get("STYLETTS2_BETA", "0.7"))
STYLETTS2_DIFFUSION_STEPS = int(os.environ.get("STYLETTS2_STEPS", "5"))
TTS_MAX_CHUNK_CHARS = int(os.environ.get("TTS_MAX_CHUNK_CHARS", "50"))

# Metrics
METRICS_FILE = os.environ.get("METRICS_FILE", "/workspace/voicebot-pipeline/call_metrics.jsonl")

# ============================================================
# Hallucination filter
# ============================================================
HALLUCINATION_PATTERNS = [
    "subscribe", "kenh", "la la school", "lala school",
    "hen gap lai cac ban", "video tiep theo",
    "dang ky kenh", "like share", "bam chuong",
    "xin chao cac ban da den voi", "cam on cac ban da xem",
    "nho dang ky", "theo doi kenh", "ung ho kenh",
    "hay subscribe", "bo lo nhung video",
]

# ============================================================
# "Im lặng" (silence/mute) mode
# ============================================================
_muted = False

MUTE_PATTERNS = [
    "im lặng", "im lang", "im đi", "im di",
    "câm", "cam",
    "đừng nói nữa", "dung noi nua",
    "không nói nữa", "khong noi nua",
    "ngừng nói", "ngung noi",
    "yên lặng", "yen lang",
]

UNMUTE_PATTERNS = [
    "nói đi", "noi di",
    "nói lại đi", "noi lai di",
    "bỏ im lặng", "bo im lang",
    "hết im lặng", "het im lang",
    "nói tiếp", "noi tiep",
    "nói lại", "noi lai",
]

# ============================================================
# Sentence / chunk splitter
# ============================================================
SENTENCE_END = re.compile(r'[.?!。!?]\s*$')


def _split_tts_chunks(text, max_chars=None):
    """Split long text into smaller chunks at comma/semicolon for even TTS pacing."""
    if max_chars is None:
        max_chars = TTS_MAX_CHUNK_CHARS
    text = text.strip()
    if not text or len(text) <= max_chars:
        return [text] if text else []

    chunks = []
    parts = re.split(r'([,;:–—]\s*)', text)
    current = ""
    for part in parts:
        if len(current) + len(part) > max_chars and current.strip():
            chunks.append(current.strip())
            current = part
        else:
            current += part
    if current.strip():
        chunks.append(current.strip())
    return chunks if chunks else [text]


# ============================================================
# Helper functions
# ============================================================

def _check_mute_command(text):
    """Returns 'mute', 'unmute', or None."""
    lower = text.lower().strip()
    for p in UNMUTE_PATTERNS:
        if p in lower:
            return "unmute"
    for p in MUTE_PATTERNS:
        if p in lower:
            return "mute"
    return None


def _is_hallucination(text):
    lower = text.lower().strip()
    if not lower or len(lower) < 3:
        return True
    for pattern in HALLUCINATION_PATTERNS:
        if pattern in lower:
            return True
    words = lower.split()
    if len(words) >= 3 and len(set(words)) == 1:
        return True
    return False


def _resample(audio_np, sr_from, sr_to):
    if sr_from == sr_to:
        return audio_np
    from scipy.signal import resample_poly
    g = gcd(sr_from, sr_to)
    return resample_poly(audio_np, sr_to // g, sr_from // g)


def noise_suppress(audio_np, sr=16000):
    df_model = _ctx.get("df_model")
    df_state = _ctx.get("df_state")
    df_enhance_fn = _ctx.get("df_enhance_fn")
    if df_model is None or df_state is None or df_enhance_fn is None:
        return audio_np
    try:
        target_sr = df_state.sr()
        audio_up = _resample(audio_np, sr, target_sr)
        audio_tensor = torch.from_numpy(audio_up).float().unsqueeze(0)
        enhanced = df_enhance_fn(df_model, df_state, audio_tensor)
        if isinstance(enhanced, torch.Tensor):
            enhanced_np = enhanced.cpu().detach().squeeze().numpy()
        else:
            enhanced_np = np.asarray(enhanced).squeeze()
        enhanced_orig = _resample(enhanced_np, target_sr, sr)
        if len(enhanced_orig) > len(audio_np):
            enhanced_orig = enhanced_orig[:len(audio_np)]
        elif len(enhanced_orig) < len(audio_np):
            enhanced_orig = np.pad(enhanced_orig, (0, len(audio_np) - len(enhanced_orig)))
        return enhanced_orig
    except Exception as e:
        print(f"[NS] DeepFilterNet error (fallback to raw audio): {e}")
        return audio_np


def estimate_snr(audio_np, sr=16000):
    frame_size = int(sr * 0.1)
    n_frames = len(audio_np) // frame_size
    if n_frames < 2:
        return 0.0
    frames = audio_np[:n_frames * frame_size].reshape(n_frames, frame_size)
    energies = np.mean(frames ** 2, axis=1)
    energies_sorted = np.sort(energies)
    n_noise = max(1, n_frames // 5)
    n_signal = max(1, n_frames // 2)
    noise_e = float(np.mean(energies_sorted[:n_noise]))
    signal_e = float(np.mean(energies_sorted[-n_signal:]))
    if noise_e < 1e-10:
        return 60.0
    return round(10 * np.log10(signal_e / noise_e), 1)


def log_metrics(metrics_dict):
    try:
        metrics_dict["timestamp"] = datetime.now(timezone.utc).isoformat()
        with open(METRICS_FILE, "a") as f:
            f.write(json.dumps(metrics_dict, ensure_ascii=False) + "\n")
    except Exception as e:
        print(f"[METRICS] Log failed: {e}")


# ============================================================
# STT
# ============================================================

def stt_transcribe(audio_bytes):
    """Transcribe audio bytes. Returns (text, metrics_dict)."""
    stt_model = _ctx["stt_model"]
    enable_deepfilter = _ctx.get("enable_deepfilter", False)

    buf = io.BytesIO(audio_bytes)
    import soundfile as sf
    audio_np, sr = sf.read(buf)
    if len(audio_np.shape) > 1:
        audio_np = audio_np.mean(axis=1)
    audio_np = audio_np.astype(np.float32)

    rms = float(np.sqrt(np.mean(audio_np ** 2)))
    snr = estimate_snr(audio_np, int(sr))
    duration = round(len(audio_np) / sr, 2)

    metrics = {
        "input_rms": round(rms, 5),
        "input_snr_est": snr,
        "input_duration_s": duration,
    }

    if rms < 0.01:
        metrics["rejected"] = "low_rms"
        return "", metrics

    if enable_deepfilter and _ctx.get("df_model") is not None:
        ns_t0 = time.time()
        audio_np = noise_suppress(audio_np, int(sr))
        metrics["ns_latency_ms"] = round((time.time() - ns_t0) * 1000, 1)

    segments, info = stt_model.transcribe(
        audio_np, language="vi",
        vad_filter=True,
        no_speech_threshold=0.7,
        vad_parameters=dict(
            min_speech_duration_ms=250,
            min_silence_duration_ms=200,
            speech_pad_ms=100,
            threshold=0.45,
        ),
    )
    seg_list = list(segments)
    text = " ".join([s.text for s in seg_list]).strip()

    if seg_list:
        try:
            avg_lp = [getattr(s, 'avg_log_prob', None) or getattr(s, 'avg_logprob', 0) for s in seg_list]
            metrics["stt_confidence"] = round(float(np.mean(avg_lp)), 3)
        except Exception:
            pass

    metrics["stt_text_len"] = len(text)
    hallucinated = _is_hallucination(text)
    metrics["hallucination"] = hallucinated
    return ("" if hallucinated else text), metrics


# ============================================================
# LLM
# ============================================================

def llm_generate_stream(user_text, conversation=None):
    """Stream tokens from local Qwen model."""
    from transformers import TextIteratorStreamer

    llm_model = _ctx["llm_model"]
    llm_tokenizer = _ctx["llm_tokenizer"]
    device = _ctx["device"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation:
        messages.extend(conversation)
    messages.append({"role": "user", "content": user_text})

    text = llm_tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    inputs = llm_tokenizer(text, return_tensors="pt").to(device)

    streamer = TextIteratorStreamer(llm_tokenizer, skip_prompt=True, skip_special_tokens=True)
    gen_kwargs = {
        **inputs,
        "streamer": streamer,
        "max_new_tokens": LLM_MAX_TOKENS,
        "temperature": LLM_TEMPERATURE,
        "top_p": 0.8,
        "top_k": 50,
        "do_sample": True,
        "repetition_penalty": 1.1,
    }

    gen_error = [None]

    def _generate():
        try:
            llm_model.generate(**gen_kwargs)
        except Exception as e:
            gen_error[0] = e

    thread = threading.Thread(target=_generate, daemon=True)
    thread.start()
    for token_text in streamer:
        if token_text:
            yield token_text
    thread.join(timeout=60)
    if gen_error[0] is not None:
        raise RuntimeError(f"LLM generation failed: {gen_error[0]}")
    if thread.is_alive():
        print("[LLM] WARNING: generation thread timed out (60s)")
        raise RuntimeError("LLM generation timed out")


def llm_generate_api(user_text, conversation=None):
    """Generate via OpenAI API."""
    llm_client = _ctx["llm_client"]
    llm_model_name = _ctx["llm_model_name"]

    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    if conversation:
        messages.extend(conversation)
    messages.append({"role": "user", "content": user_text})

    response = llm_client.chat.completions.create(
        model=llm_model_name,
        messages=messages,
        max_tokens=LLM_MAX_TOKENS,
        temperature=LLM_TEMPERATURE,
    )
    return response.choices[0].message.content.strip()


# ============================================================
# TTS
# ============================================================

@torch.inference_mode()
def _tts_synthesize_single(text):
    """Synthesize a single short chunk."""
    styletts2_model = _ctx.get("styletts2_model")
    tts_method = _ctx.get("tts_method")
    tts_styles = _ctx.get("tts_styles")

    text = text.strip()
    if not text or styletts2_model is None:
        return None

    try:
        if tts_method == "pip":
            wav = styletts2_model.inference(
                text,
                alpha=STYLETTS2_ALPHA,
                beta=STYLETTS2_BETA,
                diffusion_steps=STYLETTS2_DIFFUSION_STEPS,
                embedding_scale=1,
            )
        else:
            if tts_styles:
                with torch.no_grad():
                    wav = styletts2_model.generate(text, tts_styles)
                if isinstance(wav, np.ndarray):
                    wav = wav / max(np.abs(wav).max(), 1e-6)
            else:
                print("[TTS] No styles — cannot synthesize")
                return None
    except Exception as e:
        print(f"[TTS] Synthesis error: {e}")
        import traceback
        traceback.print_exc()
        return None

    if isinstance(wav, torch.Tensor):
        wav = wav.cpu().numpy()
    if isinstance(wav, list):
        wav = np.array(wav, dtype=np.float32)
    if wav.ndim > 1:
        wav = wav.squeeze()
    return wav


def tts_synthesize(text):
    """Synthesize text → waveform. Auto-splits long text into chunks."""
    tts_sample_rate = _ctx.get("tts_sample_rate", 24000)
    chunks = _split_tts_chunks(text)
    if not chunks:
        return None

    waveforms = []
    for chunk in chunks:
        wav = _tts_synthesize_single(chunk)
        if wav is not None:
            waveforms.append(wav)
            if len(chunks) > 1:
                waveforms.append(np.zeros(int(tts_sample_rate * 0.05), dtype=np.float32))

    if not waveforms:
        return None
    if len(chunks) > 1 and len(waveforms) > 1:
        waveforms.pop()
    return np.concatenate(waveforms)


def waveform_to_wav_bytes(waveform):
    tts_sample_rate = _ctx.get("tts_sample_rate", 24000)
    audio_int16 = np.clip(waveform * 32767, -32768, 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(tts_sample_rate)
        wf.writeframes(audio_int16.tobytes())
    return buf.getvalue()


# ============================================================
# Streaming pipeline: LLM → TTS per sentence
# ============================================================

def chat_streaming_pipeline(transcript, conversation=None):
    """Returns (full_response, wav_bytes, latency_dict)."""
    llm_t0 = time.time()
    sentence_buffer = ""
    full_response = ""
    audio_waveforms = []
    tts_time_total = 0.0
    first_sentence_time = None

    for token in llm_generate_stream(transcript, conversation):
        sentence_buffer += token
        full_response += token

        if SENTENCE_END.search(sentence_buffer):
            sentence = sentence_buffer.strip()
            sentence_buffer = ""

            if first_sentence_time is None:
                first_sentence_time = time.time() - llm_t0

            tts_t0 = time.time()
            waveform = tts_synthesize(sentence)
            tts_time_total += time.time() - tts_t0
            if waveform is not None:
                audio_waveforms.append(waveform)

    llm_time = time.time() - llm_t0 - tts_time_total

    if sentence_buffer.strip():
        tts_t0 = time.time()
        waveform = tts_synthesize(sentence_buffer.strip())
        tts_time_total += time.time() - tts_t0
        if waveform is not None:
            audio_waveforms.append(waveform)

    if audio_waveforms:
        combined = np.concatenate(audio_waveforms)
        wav_bytes = waveform_to_wav_bytes(combined)
    else:
        wav_bytes = None

    return full_response, wav_bytes, {
        "llm": round(llm_time, 2),
        "tts": round(tts_time_total, 2),
        "first_sentence": round(first_sentence_time, 2) if first_sentence_time else 0,
    }


# ============================================================
# Route handlers (called by Flask routes in server.py)
# ============================================================

def handle_chat(audio_b64, conversation):
    """Handle /chat request. Returns (response_dict, status_code)."""
    global _muted
    llm_mode = _ctx.get("llm_mode", "local")
    llm_model = _ctx.get("llm_model")
    tts_sample_rate = _ctx.get("tts_sample_rate", 24000)

    try:
        audio_bytes = base64.b64decode(audio_b64, validate=True)
    except Exception:
        return {"error": "Invalid base64 audio"}, 400

    total_start = time.time()

    # STT
    try:
        t0 = time.time()
        transcript, stt_metrics = stt_transcribe(audio_bytes)
        stt_time = time.time() - t0
        print(f"[STT] {stt_time:.2f}s: {transcript} | SNR={stt_metrics.get('input_snr_est', '?')}dB")
        if not transcript:
            return {"error": "Empty transcript"}, 200
    except Exception as e:
        print(f"[STT] Error: {e}")
        return {"error": f"STT failed: {e}"}, 500

    # Mute/unmute
    mute_cmd = _check_mute_command(transcript)
    if mute_cmd == "mute":
        _muted = True
        print(f"[MUTE] Muted by user: '{transcript}'")
        return {
            "transcript": transcript,
            "response": "Dạ, em sẽ im lặng ạ.",
            "audio_b64": "",
            "sample_rate": tts_sample_rate,
            "latency": {"stt": round(stt_time, 2), "llm": 0, "tts": 0, "total": round(time.time() - total_start, 2)},
            "muted": True,
        }, 200
    elif mute_cmd == "unmute":
        _muted = False
        print(f"[MUTE] Unmuted by user: '{transcript}'")

    # LLM + TTS
    try:
        if llm_mode == "local" and llm_model is not None:
            reply, wav_bytes, latency = chat_streaming_pipeline(transcript, conversation)
            llm_time = latency["llm"]
            tts_time = latency["tts"]
            print(f"[LLM+TTS] Streaming: LLM {llm_time:.2f}s + TTS {tts_time:.2f}s | first_sentence {latency['first_sentence']:.2f}s")
        else:
            t0 = time.time()
            reply = llm_generate_api(transcript, conversation)
            llm_time = time.time() - t0
            print(f"[LLM] {llm_time:.2f}s: {reply}")
            t0 = time.time()
            waveform = tts_synthesize(reply)
            tts_time = time.time() - t0
            wav_bytes = waveform_to_wav_bytes(waveform) if waveform is not None else None

        if _muted:
            print(f"[MUTE] Muted — skipping audio for: {reply[:80]}")
            wav_bytes = None

        print(f"[LLM] {llm_time:.2f}s: {reply[:80]}")
        audio_out_b64 = base64.b64encode(wav_bytes).decode() if wav_bytes else ""
    except Exception as e:
        print(f"[LLM/TTS] Error: {e}")
        import traceback
        traceback.print_exc()
        return {"error": f"LLM/TTS failed: {e}"}, 500

    total = time.time() - total_start
    print(f"[TOTAL] {total:.2f}s | STT {stt_time:.2f} + LLM {llm_time:.2f} + TTS {tts_time:.2f}")

    log_metrics({
        **stt_metrics,
        "stt_latency_ms": round(stt_time * 1000, 1),
        "llm_latency_ms": round(llm_time * 1000, 1),
        "tts_latency_ms": round(tts_time * 1000, 1),
        "total_latency_ms": round(total * 1000, 1),
        "transcript": transcript,
        "response_len": len(reply),
    })

    return {
        "transcript": transcript,
        "response": reply,
        "audio_b64": audio_out_b64,
        "sample_rate": tts_sample_rate,
        "muted": _muted,
        "latency": {
            "stt": round(stt_time, 2),
            "llm": round(llm_time, 2),
            "tts": round(tts_time, 2),
            "total": round(total, 2),
        },
        "metrics": {
            "input_snr": stt_metrics.get("input_snr_est"),
            "stt_confidence": stt_metrics.get("stt_confidence"),
            "noise_suppressed": stt_metrics.get("ns_latency_ms") is not None,
        },
    }, 200


def handle_ws_process(ws, audio_bytes, conversation, cancel_event=None, turn_id=0):
    """WebSocket handler: STT → streaming LLM → streaming TTS chunks."""
    import struct
    global _muted
    llm_mode = _ctx.get("llm_mode", "local")
    llm_model = _ctx.get("llm_model")

    total_start = time.time()
    turn_id_bytes = struct.pack('>I', turn_id)

    def _is_cancelled():
        return cancel_event and cancel_event.is_set()

    def _send_json(obj):
        if _is_cancelled():
            return
        obj["turn_id"] = turn_id
        ws.send(json.dumps(obj))

    def _send_audio(wav_bytes):
        if _is_cancelled():
            return
        ws.send(turn_id_bytes + wav_bytes)

    # STT
    try:
        t0 = time.time()
        transcript, stt_metrics = stt_transcribe(audio_bytes)
        stt_time = time.time() - t0
        print(f"[WS-STT] {stt_time:.2f}s: {transcript} | SNR={stt_metrics.get('input_snr_est', '?')}dB")

        if not transcript:
            _send_json({"type": "error", "error": "Empty transcript"})
            return

        _send_json({"type": "transcript", "text": transcript})
    except Exception as e:
        _send_json({"type": "error", "error": f"STT failed: {e}"})
        return

    # Mute/unmute
    mute_cmd = _check_mute_command(transcript)
    if mute_cmd == "mute":
        _muted = True
        print(f"[WS-MUTE] Muted by user: '{transcript}'")
        _send_json({
            "type": "response",
            "text": "Dạ, em sẽ im lặng ạ.",
            "muted": True,
            "latency": {"stt": round(stt_time, 2), "llm": 0, "tts": 0, "total": round(time.time() - total_start, 2)},
        })
        return
    elif mute_cmd == "unmute":
        _muted = False
        print(f"[WS-MUTE] Unmuted by user: '{transcript}'")

    if _is_cancelled():
        print(f"[WS] Turn {turn_id} cancelled after STT")
        return

    # LLM + TTS
    try:
        if llm_mode == "local" and llm_model is not None:
            llm_t0 = time.time()
            sentence_buffer = ""
            full_response = ""
            tts_time_total = 0.0

            _cancelled = False
            for token in llm_generate_stream(transcript, conversation):
                if _is_cancelled():
                    print(f"[WS] Turn {turn_id} barge-in — stopping LLM")
                    _cancelled = True
                    break
                sentence_buffer += token
                full_response += token

                if SENTENCE_END.search(sentence_buffer):
                    sentence = sentence_buffer.strip()
                    sentence_buffer = ""

                    if not _muted:
                        tts_t0 = time.time()
                        waveform = tts_synthesize(sentence)
                        tts_time_total += time.time() - tts_t0
                        if waveform is not None:
                            _send_audio(waveform_to_wav_bytes(waveform))

            llm_time = time.time() - llm_t0 - tts_time_total

            if not _cancelled and sentence_buffer.strip() and not _muted:
                tts_t0 = time.time()
                waveform = tts_synthesize(sentence_buffer.strip())
                tts_time_total += time.time() - tts_t0
                if waveform is not None:
                    _send_audio(waveform_to_wav_bytes(waveform))

            reply = full_response
        else:
            t0 = time.time()
            reply = llm_generate_api(transcript, conversation)
            llm_time = time.time() - t0

            if _is_cancelled():
                print(f"[WS] Turn {turn_id} cancelled after LLM API")
                return

            if not _muted:
                t0 = time.time()
                waveform = tts_synthesize(reply)
                tts_time_total = time.time() - t0
                if waveform is not None:
                    _send_audio(waveform_to_wav_bytes(waveform))

        total = time.time() - total_start
        print(f"[WS] Turn {turn_id} | {total:.2f}s | STT {stt_time:.2f} + LLM {llm_time:.2f} + TTS {tts_time_total:.2f}")
        print(f"  User: {transcript}")
        print(f"  Bot:  {reply[:80]}")

        log_metrics({
            **stt_metrics,
            "turn_id": turn_id,
            "cancelled": _cancelled if llm_mode == "local" else False,
            "stt_latency_ms": round(stt_time * 1000, 1),
            "llm_latency_ms": round(llm_time * 1000, 1),
            "tts_latency_ms": round(tts_time_total * 1000, 1),
            "total_latency_ms": round(total * 1000, 1),
            "transcript": transcript,
            "response_len": len(reply),
            "channel": "websocket",
        })

        if not _is_cancelled():
            _send_json({
                "type": "response",
                "text": reply,
                "muted": _muted,
                "latency": {
                    "stt": round(stt_time, 2),
                    "llm": round(llm_time, 2),
                    "tts": round(tts_time_total, 2),
                    "total": round(total, 2),
                },
                "metrics": {
                    "input_snr": stt_metrics.get("input_snr_est"),
                    "stt_confidence": stt_metrics.get("stt_confidence"),
                    "noise_suppressed": stt_metrics.get("ns_latency_ms") is not None,
                },
            })
        else:
            print(f"[WS] Turn {turn_id} cancelled — response not sent")
    except Exception as e:
        print(f"[WS] Error: {e}")
        import traceback
        traceback.print_exc()
        _send_json({"type": "error", "error": f"Processing failed: {e}"})
