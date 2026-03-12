# ============================================================
# Voicebot All-in-One — Docker Image (Groq mode)
# STT: Faster-Whisper large-v3 (GPU)
# LLM: Groq API — llama-3.3-70b-versatile (no local GPU needed)
# TTS: F5-TTS-Vietnamese-ViVoice (GPU)
#
# Image: ghcr.io/ailoveyou89/voicebot-pipeline:latest
#
# Required at runtime:
#   docker run -e GROQ_API_KEY=gsk_xxx ...
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
    huggingface_hub

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

# --- Env defaults ---
ENV PORT=5300 \
    LLM_MODE=groq \
    GROQ_MODEL=llama-3.3-70b-versatile \
    F5_TTS_REPO_DIR=/app/F5-TTS-Vietnamese \
    F5_TTS_CKPT=/app/F5-TTS-Vietnamese-ViVoice/model_last.pt \
    F5_TTS_VOCAB=/app/F5-TTS-Vietnamese/vocab.txt \
    F5_TTS_REF_AUDIO=/app/F5-TTS-Vietnamese/ref.wav \
    F5_TTS_REF_TEXT="cả hai bên hãy cố gắng hiểu cho nhau" \
    ENABLE_DEEPFILTER=0

EXPOSE 5300

CMD ["python", "-u", "all_in_one_server.py"]
