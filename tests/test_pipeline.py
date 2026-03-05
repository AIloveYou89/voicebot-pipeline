"""End-to-end pipeline test (requires GPU).

Usage:
    # Smoke test (generates silent audio)
    python -m tests.test_pipeline

    # With real audio file
    python -m tests.test_pipeline path/to/test.wav
"""
import sys
import io
import base64
import json
import time

import numpy as np
import soundfile as sf


def _make_test_audio() -> bytes:
    """Generate a short silent WAV for smoke testing."""
    sr = 16000
    audio = np.zeros(int(sr * 2), dtype=np.float32)
    buf = io.BytesIO()
    sf.write(buf, audio, sr, format="WAV")
    return buf.getvalue()


def test_pipeline_smoke():
    """Test full pipeline with silent audio (transcript will be empty → error expected)."""
    from src.pipeline import process_turn

    audio_bytes = _make_test_audio()
    result = process_turn(audio_bytes, conversation=[])

    assert "request_id" in result
    assert "status" in result
    # Silent audio → empty transcript → error is expected
    print(f"[PASS] Pipeline smoke: status={result['status']}")
    print(f"  Result: {json.dumps({k: v for k, v in result.items() if k != 'audio_b64'}, indent=2)}")


def test_pipeline_with_file(wav_path: str):
    """Test full pipeline with a real audio file."""
    from src.pipeline import process_turn

    with open(wav_path, "rb") as f:
        audio_bytes = f.read()

    print(f"[TEST] Processing {wav_path} ({len(audio_bytes)} bytes)...")
    t0 = time.time()

    result = process_turn(
        audio_bytes=audio_bytes,
        conversation=[],
        system_prompt="Ban la tro ly AI. Tra loi ngan gon bang tieng Viet.",
    )

    wall_time = time.time() - t0

    print(f"\n{'='*50}")
    print(f"Status: {result['status']}")
    print(f"Request ID: {result.get('request_id')}")

    if result["status"] == "ok":
        print(f"Transcript: {result['transcript']}")
        print(f"Response: {result['response']}")
        print(f"Latency: {json.dumps(result['latency'], indent=2)}")
        print(f"Wall time: {wall_time:.2f}s")

        # Save output audio
        audio_b64 = result.get("audio_b64")
        if audio_b64:
            out_bytes = base64.b64decode(audio_b64)
            out_path = "test_output.wav"
            with open(out_path, "wb") as f:
                f.write(out_bytes)
            print(f"Output saved: {out_path}")
    else:
        print(f"Error: stage={result.get('failed_stage')}, code={result.get('error_code')}")
        if result.get("transcript"):
            print(f"Transcript: {result['transcript']}")
        if result.get("response"):
            print(f"Response: {result['response']}")

    print(f"{'='*50}")


def test_handler_format():
    """Test RunPod handler input/output format."""
    from src.handler import handler

    audio_bytes = _make_test_audio()
    audio_b64 = base64.b64encode(audio_bytes).decode()

    job = {
        "input": {
            "audio_b64": audio_b64,
            "conversation": [],
            "system_prompt": "Tra loi ngan gon.",
        }
    }

    result = handler(job)
    assert isinstance(result, dict)
    assert "status" in result
    print(f"[PASS] Handler format: status={result['status']}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        test_pipeline_with_file(sys.argv[1])
    else:
        print("Running smoke tests (no real audio)...")
        test_pipeline_smoke()
