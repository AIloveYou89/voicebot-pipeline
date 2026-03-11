#!/bin/bash
# ==============================================
# Voicebot All-in-One — Startup Script
# Docker image has all deps pre-installed.
# Only downloads models if not found on network volume.
#
# Usage:
#   bash /app/start.sh          (from Docker image)
#   bash /workspace/voicebot-pipeline/start.sh  (from volume)
# ==============================================

set -uo pipefail

LOG_FILE="/workspace/voicebot-pipeline/server.log"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
F5_TTS_MODEL_DIR="/workspace/F5-TTS-Vietnamese-ViVoice"
F5_TTS_REPO="${F5_TTS_REPO_DIR:-/opt/F5-TTS-Vietnamese}"

export HF_HOME PORT="${PORT:-5300}"
mkdir -p "$(dirname "$LOG_FILE")" "$HF_HOME"

trap 'echo "[ERROR] FAILED at line $LINENO (exit code $?)" | tee -a "$LOG_FILE"' ERR

log() { echo "[$(date +%H:%M:%S)] $1" | tee -a "$LOG_FILE"; }

log "=============================================="
log "  Voicebot All-in-One Startup"
log "  Port: $PORT | Time: $(date)"
log "  TTS: F5-TTS-Vietnamese-ViVoice"
log "=============================================="

# --- Step 1: Download models only if missing on volume ---

# STT: Faster-Whisper large-v3
if [ ! -d "$HF_HOME/hub/models--Systran--faster-whisper-large-v3" ]; then
    log "Downloading Whisper large-v3 (~3GB)..."
    python -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')" 2>&1 | tee -a "$LOG_FILE"
    log "Whisper downloaded."
else
    log "Whisper already cached."
fi

# LLM: Qwen
LLM="${LLM_MODEL:-Qwen/Qwen2.5-3B-Instruct}"
LLM_HF_DIR=$(echo "$LLM" | tr '/' '--')
if [ ! -d "$HF_HOME/hub/models--${LLM_HF_DIR}" ]; then
    log "Downloading $LLM..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('$LLM')" 2>&1 | tee -a "$LOG_FILE"
    log "$LLM downloaded."
else
    log "$LLM already cached."
fi

# TTS: F5-TTS Vietnamese ViVoice
if [ ! -d "$F5_TTS_MODEL_DIR" ] || [ -z "$(ls -A "$F5_TTS_MODEL_DIR" 2>/dev/null)" ]; then
    log "Downloading F5-TTS-Vietnamese-ViVoice (~5GB)..."
    python -c "from huggingface_hub import snapshot_download; snapshot_download('hynt/F5-TTS-Vietnamese-ViVoice', local_dir='$F5_TTS_MODEL_DIR')" 2>&1 | tee -a "$LOG_FILE"
    # Rename config.json to vocab.txt (required by F5-TTS)
    if [ -f "$F5_TTS_MODEL_DIR/config.json" ] && [ ! -f "$F5_TTS_MODEL_DIR/vocab.txt" ]; then
        mv "$F5_TTS_MODEL_DIR/config.json" "$F5_TTS_MODEL_DIR/vocab.txt"
        log "Renamed config.json → vocab.txt"
    fi
    log "F5-TTS model downloaded."
else
    log "F5-TTS model already cached."
fi

# --- Step 2: Set F5-TTS env ---
export F5_TTS_REPO_DIR="$F5_TTS_REPO"
export F5_TTS_CKPT="$F5_TTS_MODEL_DIR/model_last.pt"
export F5_TTS_VOCAB="$F5_TTS_REPO/vocab.txt"
export F5_TTS_REF_AUDIO="${F5_TTS_REF_AUDIO:-$F5_TTS_REPO/ref.wav}"
export F5_TTS_REF_TEXT="${F5_TTS_REF_TEXT:-cả hai bên hãy cố gắng hiểu cho nhau}"

# --- Step 3: Copy models to local SSD for faster loading ---
LOCAL_HF="/tmp/huggingface"
if [ ! -d "$LOCAL_HF/hub" ]; then
    log "Copying models NFS → local SSD..."
    t_start=$(date +%s)
    mkdir -p "$LOCAL_HF"
    cp -r "$HF_HOME/hub" "$LOCAL_HF/hub" 2>/dev/null || true
    [ -d "$HF_HOME/modules" ] && cp -r "$HF_HOME/modules" "$LOCAL_HF/modules" 2>/dev/null || true
    t_end=$(date +%s)
    log "Models copied in $((t_end - t_start))s"
else
    log "Models already on local SSD."
fi
export HF_HOME="$LOCAL_HF"

# --- Step 4: Use workspace code if available ---
if [ -d "/workspace/voicebot-pipeline" ] && [ -f "/workspace/voicebot-pipeline/all_in_one_server.py" ]; then
    cd /workspace/voicebot-pipeline
    log "Using code from /workspace/voicebot-pipeline"
else
    cd /app
    log "Using code from /app (no workspace code found)"
fi

# --- Step 5: Smoke test ---
log "Verifying imports..."
python -c "
import torch, numpy
from faster_whisper import WhisperModel
print(f'torch={torch.__version__} cuda={torch.cuda.is_available()} numpy={numpy.__version__}')
try:
    from f5_tts.api import F5TTS
    print('f5-tts OK')
except ImportError:
    print('f5-tts NOT available')
try:
    from df.enhance import enhance, init_df
    print('deepfilternet OK')
except ImportError:
    print('deepfilternet NOT available')
" 2>&1 | tee -a "$LOG_FILE"

# --- Step 6: Start server ---
log "Launching all_in_one_server.py on port $PORT..."
exec python -u all_in_one_server.py 2>&1 | tee -a "$LOG_FILE"
