# ============================================================
# Voicebot All-in-One — Docker Image
# Pre-bakes: OS deps, pip packages, model weights
# Runtime: load models vào GPU (~30s thay vì ~5min download+load)
#
# Image: ghcr.io/ailoveyou89/voicebot-pipeline:latest
# ============================================================

FROM pytorch/pytorch:2.4.0-cuda12.4-cudnn9-runtime

WORKDIR /app

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg espeak-ng git \
    && rm -rf /var/lib/apt/lists/*

# --- Python deps (cached layer) ---
COPY constraints.txt .
RUN pip install --no-cache-dir \
    -c constraints.txt \
    "numpy>=2.0,<3" setuptools wheel packaging \
    flask flask-cors flask-sock faster-whisper soundfile \
    huggingface_hub openai scipy \
    styletts2 phonemizer

# --- Download models into image ---
# STT: Faster-Whisper large-v3 (~3GB)
RUN python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"

# LLM: Qwen2.5-7B-Instruct (~15GB)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('Qwen/Qwen2.5-7B-Instruct')"

# TTS: StyleTTS2-lite-vi (~200MB)
RUN python -c "from huggingface_hub import snapshot_download; snapshot_download('dangtr0408/StyleTTS2-lite-vi', local_dir='/app/styletts2_lite_vi')"

# --- Copy app code (small layer — fast rebuild on code changes) ---
COPY all_in_one_server.py handlers.py start.sh ./

# --- Env defaults ---
ENV PORT=5300 \
    LLM_MODE=local \
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
    STYLETTS2_MODEL_DIR=/app/styletts2_lite_vi \
    ENABLE_DEEPFILTER=0

EXPOSE 5300

CMD ["python", "-u", "all_in_one_server.py"]
