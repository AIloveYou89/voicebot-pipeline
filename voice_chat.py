"""
Real-time voice chat with voicebot pipeline.
Speak naturally → bot listens → responds with voice → loops.

Usage:
    pip install sounddevice numpy requests
    python voice_chat.py

Controls:
    - Just speak! Bot detects when you start/stop talking.
    - Ctrl+C to quit.
"""
import io
import os
import sys
import json
import time
import wave
import base64
import struct
import threading
from collections import deque

import numpy as np
import requests

# ============================================================
# Config
# ============================================================
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY", "")
ENDPOINT_ID = os.getenv("RUNPOD_ENDPOINT_ID", "5wffpipznb10s8")
RUNPOD_URL = f"https://api.runpod.ai/v2/{ENDPOINT_ID}"

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_SIZE = 1600  # 100ms blocks at 16kHz

# VAD settings
ENERGY_THRESHOLD = 0.02       # Minimum RMS energy to detect speech
SILENCE_DURATION = 1.2        # Seconds of silence to stop recording
MIN_SPEECH_DURATION = 0.5     # Minimum speech duration (ignore short noises)

# Conversation history
conversation = []
system_prompt = None


def load_api_key():
    """Load RunPod API key from env file."""
    global RUNPOD_API_KEY
    if RUNPOD_API_KEY:
        return

    env_file = os.path.expanduser("~/.env.agentic")
    if os.path.exists(env_file):
        with open(env_file) as f:
            for line in f:
                if line.startswith("RUNPOD_API_KEY="):
                    RUNPOD_API_KEY = line.strip().split("=", 1)[1]
                    return

    print("ERROR: RUNPOD_API_KEY not found. Set it in ~/.env.agentic or as env var.")
    sys.exit(1)


def audio_to_wav_b64(audio_frames: list[np.ndarray]) -> str:
    """Convert recorded audio frames to base64 WAV."""
    audio = np.concatenate(audio_frames)
    # Normalize to int16
    audio_int16 = (audio * 32767).astype(np.int16)

    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(CHANNELS)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(SAMPLE_RATE)
        wf.writeframes(audio_int16.tobytes())

    return base64.b64encode(buf.getvalue()).decode()


def call_voicebot(audio_b64: str) -> dict:
    """Send audio to RunPod endpoint and get streaming response."""
    headers = {
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "input": {
            "audio_b64": audio_b64,
            "stream": True,
            "conversation": conversation[-6:],  # Last 3 turns
            "system_prompt": system_prompt,
        }
    }

    # Submit job
    resp = requests.post(f"{RUNPOD_URL}/run", json=payload, headers=headers)
    resp.raise_for_status()
    job_id = resp.json()["id"]

    # Poll for results
    while True:
        resp = requests.get(f"{RUNPOD_URL}/stream/{job_id}", headers=headers)
        resp.raise_for_status()
        data = resp.json()

        if data.get("stream"):
            return data

        status = data.get("status", "UNKNOWN")
        if status in ("FAILED", "CANCELLED", "TIMED_OUT"):
            print(f"\n  [ERROR] Job {status}")
            return None

        time.sleep(0.3)


def play_audio_chunks(stream_data: list) -> str:
    """Play audio chunks and return full response text."""
    import sounddevice as sd

    full_response = ""

    for chunk in stream_data:
        output = chunk.get("output", chunk)
        if not isinstance(output, dict):
            continue

        chunk_type = output.get("type")

        if chunk_type == "transcript":
            text = output.get("text", "")
            print(f"\n  🎤 Bạn: {text}")

        elif chunk_type == "audio_chunk":
            text = output.get("text", "")
            audio_b64 = output.get("audio_b64", "")

            if audio_b64:
                # Decode and play
                audio_bytes = base64.b64decode(audio_b64)
                buf = io.BytesIO(audio_bytes)
                with wave.open(buf, 'rb') as wf:
                    sr = wf.getframerate()
                    frames = wf.readframes(wf.getnframes())
                    audio_np = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32767.0

                # Play this chunk
                sd.play(audio_np, sr)
                sd.wait()

        elif chunk_type == "done":
            full_response = output.get("response", "")
            latency = output.get("latency", {})
            print(f"  🤖 Bot: {full_response}")
            print(f"     ⏱️  STT={latency.get('stt','?')}s | LLM={latency.get('llm_first_phrase','?')}s | TTS={latency.get('tts_total','?')}s | Total={latency.get('total','?')}s")

    return full_response


def listen_and_detect() -> list[np.ndarray] | None:
    """
    Listen to microphone, detect speech, record until silence.
    Returns list of audio frames, or None if interrupted.
    """
    import sounddevice as sd

    frames = []
    is_speaking = False
    silence_blocks = 0
    speech_blocks = 0
    silence_threshold = int(SILENCE_DURATION / (BLOCK_SIZE / SAMPLE_RATE))
    min_speech_blocks = int(MIN_SPEECH_DURATION / (BLOCK_SIZE / SAMPLE_RATE))

    print("\n  🎧 Đang nghe... (nói gì đi!)", end="", flush=True)

    def callback(indata, frame_count, time_info, status):
        nonlocal is_speaking, silence_blocks, speech_blocks

        audio = indata[:, 0].copy()
        rms = np.sqrt(np.mean(audio ** 2))

        if rms > ENERGY_THRESHOLD:
            if not is_speaking:
                is_speaking = True
                print("\r  🔴 Đang ghi âm...                ", end="", flush=True)
            silence_blocks = 0
            speech_blocks += 1
            frames.append(audio)
        elif is_speaking:
            silence_blocks += 1
            frames.append(audio)  # Keep recording during brief pauses

            if silence_blocks >= silence_threshold:
                raise sd.CallbackStop()
        # Not speaking yet — don't record

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=CHANNELS,
            blocksize=BLOCK_SIZE,
            dtype='float32',
            callback=callback,
        ):
            # Block until callback raises CallbackStop
            while True:
                sd.sleep(100)
    except sd.CallbackStop:
        pass

    if speech_blocks < min_speech_blocks:
        # Too short — probably noise
        return None

    return frames if frames else None


def main():
    load_api_key()

    # Check sounddevice
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        default_input = sd.query_devices(kind='input')
        print(f"🎙️  Mic: {default_input['name']}")
    except Exception as e:
        print(f"ERROR: sounddevice not working: {e}")
        print("Install: pip install sounddevice")
        sys.exit(1)

    print("=" * 50)
    print("  🤖 VOICEBOT — Real-time Voice Chat")
    print("=" * 50)
    print("  Nói chuyện tự nhiên, bot sẽ nghe và trả lời.")
    print("  Nhấn Ctrl+C để thoát.\n")

    global conversation

    while True:
        try:
            # Step 1: Listen for speech
            frames = listen_and_detect()

            if frames is None:
                continue

            # Step 2: Convert to base64
            duration = len(frames) * BLOCK_SIZE / SAMPLE_RATE
            print(f"\r  📤 Đang gửi ({duration:.1f}s audio)...           ", flush=True)
            audio_b64 = audio_to_wav_b64(frames)

            # Step 3: Call voicebot API
            t0 = time.time()
            result = call_voicebot(audio_b64)

            if result is None:
                print("  ❌ Lỗi, thử lại...")
                continue

            # Step 4: Play response
            stream_chunks = result.get("stream", [])
            full_response = play_audio_chunks(stream_chunks)

            # Step 5: Update conversation history
            # Find transcript from chunks
            transcript = ""
            for chunk in stream_chunks:
                output = chunk.get("output", chunk)
                if isinstance(output, dict) and output.get("type") == "transcript":
                    transcript = output.get("text", "")
                    break

            if transcript and full_response:
                conversation.append({"role": "user", "content": transcript})
                conversation.append({"role": "assistant", "content": full_response})

        except KeyboardInterrupt:
            print("\n\n  👋 Tạm biệt!")
            break
        except Exception as e:
            print(f"\n  ❌ Error: {e}")
            continue


if __name__ == "__main__":
    main()
