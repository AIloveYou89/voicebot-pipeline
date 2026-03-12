# ============================================================
# Voicebot All-in-One — Docker Image
# Pre-bakes: OS deps, pip packages, model weights
# Runtime: load models vào GPU (~30s thay vì ~5min download+load)
#
# Image: ghcr.io/ailoveyou89/voicebot-pipeline:latest
#
# LLM modes:
#   LLM_MODE=groq  → Groq API (recommended, no GPU contention)
#   LLM_MODE=local → Qwen2.5 on GPU (needs more VRAM)
# ============================================================

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps (cached layer) ---
COPY constraints.txt requirements.txt ./
RUN pip install --no-cache-dir \
    -c constraints.txt \
    -r requirements.txt \
    transformers accelerate huggingface_hub

# --- F5-TTS Vietnamese (local fork) ---
RUN git clone https://github.com/nguyenthienhy/F5-TTS-Vietnamese /app/F5-TTS-Vietnamese \
    && pip install --no-cache-dir -e /app/F5-TTS-Vietnamese

# --- Download models into image ---
# STT: Faster-Whisper large-v3 (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# TTS: F5-TTS-Vietnamese-ViVoice (~5GB)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('hynt/F5-TTS-Vietnamese-ViVoice', local_dir='/app/F5-TTS-Vietnamese-ViVoice')" \
    && mv /app/F5-TTS-Vietnamese-ViVoice/config.json /app/F5-TTS-Vietnamese-ViVoice/vocab.txt

# --- Copy app code (small layer — fast rebuild on code changes) ---
COPY all_in_one_server.py handlers.py tools.py quality_metrics.py voice_web.py requirements.txt start.sh ./

# --- Env defaults (Groq mode — no local LLM needed) ---
ENV PORT=5300 \
    LLM_MODE=groq \
    GROQ_MODEL=llama-3.3-70b-versatile \
    F5_TTS_REPO_DIR=/app/F5-TTS-Vietnamese \
    F5_TTS_CKPT=/app/F5-TTS-Vietnamese-ViVoice/model_last.pt \
    F5_TTS_VOCAB=/app/F5-TTS-Vietnamese/vocab.txt \
    F5_TTS_REF_AUDIO=/app/F5-TTS-Vietnamese/ref.wav \
    F5_TTS_REF_TEXT="cả hai bên hãy cố gắng hiểu cho nhau" \
    ENABLE_DEEPFILTER=0

# GROQ_API_KEY must be set at runtime:
#   docker run -e GROQ_API_KEY=gsk_xxx ...
# Or for local LLM mode:
#   docker run -e LLM_MODE=local -e LLM_MODEL=Qwen/Qwen2.5-3B-Instruct ...

EXPOSE 5300

CMD ["python", "-u", "all_in_one_server.py"]
