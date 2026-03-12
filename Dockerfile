# ============================================================
# Voicebot All-in-One — Docker Image (deps + tools only)
# Models: network volume /workspace (NOT baked into image)
# Claude Code: pre-installed for on-pod editing
# Image: ghcr.io/ailoveyou89/voicebot-pipeline:latest
# ============================================================

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

# --- OS deps ---
RUN apt-get update && apt-get install -y --no-install-recommends \
    libsndfile1 ffmpeg git curl xz-utils \
    && rm -rf /var/lib/apt/lists/*

# --- Node.js 20 (direct binary — nodesource fails in some DCs) ---
RUN curl -fsSL https://nodejs.org/dist/v20.18.2/node-v20.18.2-linux-x64.tar.xz \
    | tar -xJ -C /usr/local --strip-components=1

# --- Claude Code (for editing on pod) ---
RUN npm install -g @anthropic-ai/claude-code

# --- Python deps (cached layer) ---
WORKDIR /app
COPY constraints.txt requirements.txt ./
# Force-remove distutils-installed blinker 1.4 (can't pip uninstall)
RUN rm -rf /usr/lib/python3/dist-packages/blinker* \
    && pip install --no-cache-dir -c constraints.txt -r requirements.txt \
       transformers accelerate huggingface_hub

# --- F5-TTS Vietnamese (local fork — editable install) ---
RUN git clone https://github.com/nguyenthienhy/F5-TTS-Vietnamese /opt/F5-TTS-Vietnamese \
    && pip install --no-cache-dir -e /opt/F5-TTS-Vietnamese

# --- Rust toolchain (needed by DeepFilterNet) ---
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

# --- DeepFilterNet noise suppression ---
RUN pip install --no-cache-dir deepfilternet || echo "DeepFilterNet skipped"

# --- App code (fallback if /workspace not mounted) ---
COPY all_in_one_server.py handlers.py tools.py quality_metrics.py start.sh ./
RUN chmod +x start.sh

# --- Env defaults ---
ENV PORT=5300 \
    LLM_MODE=local \
    LLM_MODEL=Qwen/Qwen2.5-7B-Instruct \
    F5_TTS_REPO_DIR=/opt/F5-TTS-Vietnamese \
    ENABLE_DEEPFILTER=0

# --- Auto-start: wrap RunPod's /start.sh to launch voicebot too ---
# RunPod base image has /start.sh that starts Jupyter + SSH.
# We rename it, create a new /start.sh that starts voicebot in background
# THEN runs RunPod's original start → Jupyter + SSH still work.
RUN if [ -f /start.sh ]; then \
      mv /start.sh /start_runpod.sh && \
      printf '#!/bin/bash\nmkdir -p /workspace/voicebot-pipeline\nnohup bash /app/start.sh >> /workspace/voicebot-pipeline/server.log 2>&1 &\nexec /start_runpod.sh "$@"\n' > /start.sh && \
      chmod +x /start.sh; \
    fi

EXPOSE 5300
