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

# Models download at first cold start (cached by RunPod FlashBoot after)
# Set HF_TOKEN as env var in RunPod endpoint settings

# RunPod serverless entrypoint
CMD ["python", "-u", "src/handler.py"]
