"""Test STT module (requires GPU)."""
import os
import time

import numpy as np
import soundfile as sf


def test_transcribe_wav():
    """Test Whisper transcription with a generated test tone."""
    from src.stt.whisper_stt import transcribe

    # Generate a short silent WAV for smoke test
    sr = 16000
    duration = 1.0
    audio = np.zeros(int(sr * duration), dtype=np.float32)

    import io
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    audio_bytes = buf.getvalue()

    text, latency = transcribe(audio_bytes)
    assert isinstance(text, str)
    assert isinstance(latency, float)
    assert latency > 0
    print(f"[PASS] STT: text='{text}', latency={latency:.2f}s")


def test_transcribe_from_file():
    """Test with actual WAV file if available."""
    test_files = [
        "prompts/consent_audio.wav",
        "/runpod-volume/workspace/consent_audio.wav",
    ]

    for path in test_files:
        if os.path.exists(path):
            with open(path, "rb") as f:
                audio_bytes = f.read()

            from src.stt.whisper_stt import transcribe
            text, latency = transcribe(audio_bytes)

            assert len(text) > 0, "Expected non-empty transcript"
            assert latency > 0
            print(f"[PASS] STT from file: '{text[:60]}...', latency={latency:.2f}s")
            return

    print("[SKIP] No test audio file found")


if __name__ == "__main__":
    test_transcribe_wav()
    test_transcribe_from_file()
