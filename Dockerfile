FROM pytorch/pytorch:2.6.0-cuda12.6-cudnn9-devel

# System deps for soundfile (libsndfile)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git build-essential \
    espeak-ng gfortran cmake \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Cache bust: v6-torchvision
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# llama-cpp-python with CUDA support (VieNeu-TTS fast mode)
ENV CMAKE_ARGS="-DGGML_CUDA=on"
RUN pip install --no-cache-dir llama-cpp-python==0.3.16

# Copy source
COPY src/ src/
RUN mkdir -p prompts

# Ensure imports work: from src.config import ...
ENV PYTHONPATH=/app

# Use Network Volume for HuggingFace model cache (avoids disk full on container)
# Mount volume at /runpod-volume in endpoint settings
ENV HF_HOME=/runpod-volume/huggingface
ENV TRANSFORMERS_CACHE=/runpod-volume/huggingface/hub

# Models auto-download on first cold start, persist on Network Volume

# Verify imports work at build time (catch errors early)
RUN python -c "import torch; print(f'torch {torch.__version__} CUDA={torch.cuda.is_available()}')" && \
    python -c "import transformers; print(f'transformers {transformers.__version__}')" && \
    python -c "import vieneu; print(f'vieneu OK')" && \
    python -c "from src.config import TTS_ENGINE; print(f'TTS_ENGINE={TTS_ENGINE}')"

# RunPod serverless entrypoint
CMD ["python", "-u", "src/handler.py"]
