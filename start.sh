#!/bin/bash
# ==============================================
# Voicebot All-in-One — Auto Startup Script
# Luu tai /workspace/voicebot-pipeline/start.sh
# Chay: bash /workspace/voicebot-pipeline/start.sh
#
# KHONG dung set -e (che loi am).
# Dung trap ERR + log tung buoc.
#
# --- TTS ENGINE ---
# F5-TTS-Vietnamese-ViVoice (1000h Vietnamese speech)
# Repo: https://github.com/nguyenthienhy/F5-TTS-Vietnamese
# Model: https://huggingface.co/hynt/F5-TTS-Vietnamese-ViVoice
#
# Noise suppression: DeepFilterNet (server-side, before Whisper)
# Disable: export ENABLE_DEEPFILTER=0
# ==============================================

set -uo pipefail

LOG_FILE="/workspace/voicebot-pipeline/server.log"

# Error trap — log dong loi thay vi chet am
trap 'echo "[ERROR] FAILED at line $LINENO (exit code $?)" | tee -a "$LOG_FILE"' ERR

cd /workspace/voicebot-pipeline

export PORT="${PORT:-5300}"
VENV_DIR="/workspace/voicebot-pipeline/.venv"
REQ_HASH_FILE="$VENV_DIR/.req_hash"
F5TTS_MARKER="$VENV_DIR/.f5tts_installed"
DEEPFILTER_MARKER="$VENV_DIR/.deepfilter_installed"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
CONSTRAINTS_FILE="/workspace/voicebot-pipeline/constraints.txt"
BASE_REQ_MARKER_INPUTS=(
    "$CONSTRAINTS_FILE"
    "/workspace/voicebot-pipeline/start.sh"
)

# F5-TTS paths
F5_TTS_REPO="/workspace/F5-TTS-Vietnamese"
F5_TTS_MODEL_DIR="/workspace/F5-TTS-Vietnamese-ViVoice"

mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
export HF_HOME XDG_CACHE_HOME PIP_CACHE_DIR

echo "==============================================" | tee -a "$LOG_FILE"
echo "  Voicebot All-in-One Startup" | tee -a "$LOG_FILE"
echo "  Port: $PORT" | tee -a "$LOG_FILE"
echo "  Time: $(date)" | tee -a "$LOG_FILE"
echo "  TTS: F5-TTS-Vietnamese-ViVoice" | tee -a "$LOG_FILE"
echo "  NS: DeepFilterNet" | tee -a "$LOG_FILE"
echo "  HF_HOME: $HF_HOME" | tee -a "$LOG_FILE"
echo "  PIP_CACHE_DIR: $PIP_CACHE_DIR" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"

# --- Step 0: Ensure OS deps exist on fresh pod/container ---
if ! command -v ffmpeg >/dev/null 2>&1 || ! dpkg -s libsndfile1 >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1; then
    echo "[SETUP] Installing OS dependencies..." | tee -a "$LOG_FILE"
    export DEBIAN_FRONTEND=noninteractive
    stdbuf -oL apt-get update 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] apt-get update failed" | tee -a "$LOG_FILE"
        exit 1
    fi

    stdbuf -oL apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] apt-get install failed" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# --- Step 1: Create venv with system-site-packages ---
VENV_VERSION="v3-f5tts"
VENV_VERSION_FILE="$VENV_DIR/.venv_version"

CURRENT_VENV_VER=$(cat "$VENV_VERSION_FILE" 2>/dev/null || echo "")
if [ "$CURRENT_VENV_VER" != "$VENV_VERSION" ]; then
    if [ -d "$VENV_DIR" ]; then
        echo "[SETUP] Venv outdated ($CURRENT_VENV_VER → $VENV_VERSION). Rebuilding..." | tee -a "$LOG_FILE"
        rm -rf "$VENV_DIR"
    fi
    echo "[SETUP] Creating virtualenv (system-site-packages)..." | tee -a "$LOG_FILE"
    python3 -m venv --system-site-packages "$VENV_DIR"
    if [ $? -ne 0 ]; then
        echo "[ERROR] Failed to create virtualenv" | tee -a "$LOG_FILE"
        exit 1
    fi
    echo "$VENV_VERSION" > "$VENV_VERSION_FILE"
    echo "[SETUP] Virtualenv created ($VENV_VERSION)." | tee -a "$LOG_FILE"
fi

# Activate venv
source "$VENV_DIR/bin/activate"

if [ ! -f "$CONSTRAINTS_FILE" ]; then
    echo "[ERROR] constraints.txt not found: $CONSTRAINTS_FILE" | tee -a "$LOG_FILE"
    exit 1
fi

echo "[SETUP] Upgrading pip/setuptools/wheel in venv..." | tee -a "$LOG_FILE"
stdbuf -oL python -m pip install --cache-dir "$PIP_CACHE_DIR" --upgrade pip setuptools wheel 2>&1 | tee -a "$LOG_FILE"
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] pip bootstrap failed" | tee -a "$LOG_FILE"
    exit 1
fi

# --- Step 2: Install base deps ---
CURRENT_HASH=$(cat "${BASE_REQ_MARKER_INPUTS[@]}" | md5sum | cut -d' ' -f1 || echo "none")
SAVED_HASH=$(cat "$REQ_HASH_FILE" 2>/dev/null || echo "")

if [ "$CURRENT_HASH" != "$SAVED_HASH" ]; then
    echo "[SETUP] Installing base dependencies..." | tee -a "$LOG_FILE"

    stdbuf -oL python -m pip install --cache-dir "$PIP_CACHE_DIR" -c "$CONSTRAINTS_FILE" \
        "numpy>=2.0,<3" setuptools wheel packaging \
        flask flask-cors flask-sock faster-whisper soundfile huggingface_hub openai scipy librosa 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] pip install (base) failed" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "$CURRENT_HASH" > "$REQ_HASH_FILE"
    echo "[SETUP] Base dependencies installed." | tee -a "$LOG_FILE"
else
    echo "[SETUP] Base dependencies already installed (cached)." | tee -a "$LOG_FILE"
fi

# --- Step 2b: Install F5-TTS Vietnamese (local fork) ---
if [ ! -f "$F5TTS_MARKER" ]; then
    echo "[SETUP] Installing F5-TTS Vietnamese..." | tee -a "$LOG_FILE"

    # Clone repo if not exists
    if [ ! -d "$F5_TTS_REPO" ]; then
        echo "[SETUP] Cloning F5-TTS-Vietnamese repo..." | tee -a "$LOG_FILE"
        git clone https://github.com/nguyenthienhy/F5-TTS-Vietnamese "$F5_TTS_REPO" 2>&1 | tee -a "$LOG_FILE"
    fi

    # Install as editable package
    stdbuf -oL python -m pip install --cache-dir "$PIP_CACHE_DIR" -e "$F5_TTS_REPO" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] pip install (f5-tts) failed" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "$(date)" > "$F5TTS_MARKER"
    echo "[SETUP] F5-TTS Vietnamese installed." | tee -a "$LOG_FILE"
else
    echo "[SETUP] F5-TTS Vietnamese already installed (cached)." | tee -a "$LOG_FILE"
fi

# --- Step 2c: Install DeepFilterNet noise suppression ---
if [ ! -f "$DEEPFILTER_MARKER" ]; then
    echo "[SETUP] Installing DeepFilterNet..." | tee -a "$LOG_FILE"

    stdbuf -oL python -m pip install --cache-dir "$PIP_CACHE_DIR" -c "$CONSTRAINTS_FILE" \
        deepfilternet 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[WARN] DeepFilterNet install failed — continuing without noise suppression" | tee -a "$LOG_FILE"
        export ENABLE_DEEPFILTER=0
    else
        echo "$(date)" > "$DEEPFILTER_MARKER"
        echo "[SETUP] DeepFilterNet installed." | tee -a "$LOG_FILE"
    fi
else
    echo "[SETUP] DeepFilterNet already installed (cached)." | tee -a "$LOG_FILE"
fi

# --- Step 2d: Download F5-TTS model from HuggingFace ---
if [ ! -d "$F5_TTS_MODEL_DIR" ] || [ -z "$(ls -A "$F5_TTS_MODEL_DIR" 2>/dev/null)" ]; then
    echo "[SETUP] Downloading F5-TTS-Vietnamese-ViVoice model..." | tee -a "$LOG_FILE"
    stdbuf -oL python -c "
from huggingface_hub import snapshot_download
snapshot_download('hynt/F5-TTS-Vietnamese-ViVoice', local_dir='$F5_TTS_MODEL_DIR')
print('Model downloaded to $F5_TTS_MODEL_DIR')
" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] F5-TTS model download failed" | tee -a "$LOG_FILE"
        exit 1
    fi

    # Rename config.json to vocab.txt (required by F5-TTS)
    if [ -f "$F5_TTS_MODEL_DIR/config.json" ] && [ ! -f "$F5_TTS_MODEL_DIR/vocab.txt" ]; then
        mv "$F5_TTS_MODEL_DIR/config.json" "$F5_TTS_MODEL_DIR/vocab.txt"
        echo "[SETUP] Renamed config.json → vocab.txt" | tee -a "$LOG_FILE"
    fi
else
    echo "[SETUP] F5-TTS model already downloaded." | tee -a "$LOG_FILE"
fi

# --- Step 3: Smoke test imports before serving ---
echo "[CHECK] Verifying imports..." | tee -a "$LOG_FILE"
stdbuf -oL python - <<'PY' 2>&1 | tee -a "$LOG_FILE"
import torch
import numpy
from faster_whisper import WhisperModel

print(f"torch={torch.__version__} cuda={torch.cuda.is_available()}")
print(f"numpy={numpy.__version__}")
print("faster-whisper OK")

try:
    from f5_tts.api import F5TTS
    print("f5-tts OK")
except ImportError:
    print("f5-tts NOT installed")

try:
    from df.enhance import enhance, init_df
    print("deepfilternet OK")
except ImportError:
    print("deepfilternet NOT installed — no noise suppression")
PY
if [ ${PIPESTATUS[0]} -ne 0 ]; then
    echo "[ERROR] import smoke test failed" | tee -a "$LOG_FILE"
    exit 1
fi

# --- Step 4: Use models directly from network volume (no SSD copy) ---
echo "[SPEED] Using models directly from $HF_HOME (no SSD copy)" | tee -a "$LOG_FILE"

# --- Step 5: Start server ---
echo "[START] Launching all_in_one_server.py on port $PORT..." | tee -a "$LOG_FILE"
stdbuf -oL python -u all_in_one_server.py 2>&1 | tee -a "$LOG_FILE"

# If we get here, server exited
EXIT_CODE=$?
echo "[EXIT] Server exited at $(date) with code $EXIT_CODE" | tee -a "$LOG_FILE"
