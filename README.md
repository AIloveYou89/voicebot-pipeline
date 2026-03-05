# Voice Bot Pipeline — All-in-One GPU Endpoint

STT (Whisper) + LLM (Qwen2.5-7B) + TTS (SparkTTS) on a single GPU.
Eliminates network round-trips between stages. Target: < 2s total latency.

## Architecture (Phase 1: HTTP sync)

```
Client (HTTP POST) ──── RunPod GPU Endpoint ────
                         ┌─────────────────────┐
  audio_b64 ──────────>  │ Whisper large-v3     │ ──> transcript
                         │ Qwen2.5-7B-AWQ      │ ──> response text
  audio_b64 <──────────  │ Vi-SparkTTS-0.5B     │ ──> audio
                         └─────────────────────┘
```

## Models & VRAM

| Model | VRAM | Purpose |
|-------|------|---------|
| Whisper large-v3 | ~3 GB | Vietnamese STT |
| Qwen2.5-7B-Instruct-AWQ | ~5 GB | LLM (INT4 quantized) |
| Vi-SparkTTS-0.5B | ~3 GB | Vietnamese TTS |
| **Total** | **~15-18 GB** | Fits RTX 4090 (24GB) |

## Quick Start

```bash
# Build Docker image
docker build --build-arg HF_TOKEN=hf_xxx -t voicebot-pipeline .

# Test locally (requires GPU)
python -m tests.test_pipeline test_audio.wav

# Deploy to RunPod
docker push yourusername/voicebot-pipeline:latest
# Create serverless endpoint on RunPod dashboard (RTX 4090)
```

## API

**Input:**
```json
{
  "audio_b64": "<base64 WAV>",
  "conversation": [{"role": "user", "content": "..."}],
  "system_prompt": "optional"
}
```

**Output (success):**
```json
{
  "status": "ok",
  "request_id": "abc12345",
  "transcript": "user speech",
  "response": "bot reply",
  "audio_b64": "<base64 WAV>",
  "latency": {"stt": 0.3, "llm": 0.4, "tts": 0.7, "total": 1.4}
}
```

## Constraints

- `torch==2.5.1` — SparkTTS breaks on 2.6+
- `transformers==4.46.2` — pinned for SparkTTS compatibility
- `processor.model = model` — NOT `link_model()`
- `torch.take` monkey-patch must be kept (int64 index fix)
