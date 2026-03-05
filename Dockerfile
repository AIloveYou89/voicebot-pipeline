FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

# System deps for soundfile (libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Install Python deps
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source
COPY src/ src/
COPY prompts/ prompts/

# Pre-download models at build time (baked into image → no cold start)
ARG HF_TOKEN
ENV HF_TOKEN=${HF_TOKEN}

RUN python -c "\
from huggingface_hub import snapshot_download; \
snapshot_download('openai/whisper-large-v3', token='${HF_TOKEN}'); \
snapshot_download('Qwen/Qwen2.5-7B-Instruct-AWQ', token='${HF_TOKEN}'); \
snapshot_download('DragonLineageAI/Vi-SparkTTS-0.5B', token='${HF_TOKEN}'); \
"

# RunPod serverless entrypoint
CMD ["python", "-u", "src/handler.py"]
