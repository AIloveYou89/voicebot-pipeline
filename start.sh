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
# v2: StyleTTS2-lite-vi (replaced Matcha-TTS for better prosody)
# Model auto-downloaded from HuggingFace: dangtr0408/StyleTTS2-lite-vi
# Override: export STYLETTS2_MODEL_DIR="/workspace/my_model"
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
STYLETTS2_MARKER="$VENV_DIR/.styletts2_installed"
DEEPFILTER_MARKER="$VENV_DIR/.deepfilter_installed"
PIP_CACHE_DIR="${PIP_CACHE_DIR:-/workspace/pip-cache}"
HF_HOME="${HF_HOME:-/workspace/huggingface}"
XDG_CACHE_HOME="${XDG_CACHE_HOME:-/workspace/.cache}"
CONSTRAINTS_FILE="/workspace/voicebot-pipeline/constraints.txt"
BASE_REQ_MARKER_INPUTS=(
    "$CONSTRAINTS_FILE"
    "/workspace/voicebot-pipeline/start.sh"
)

mkdir -p "$PIP_CACHE_DIR" "$HF_HOME" "$XDG_CACHE_HOME"
export HF_HOME XDG_CACHE_HOME PIP_CACHE_DIR

echo "==============================================" | tee -a "$LOG_FILE"
echo "  Voicebot All-in-One Startup" | tee -a "$LOG_FILE"
echo "  Port: $PORT" | tee -a "$LOG_FILE"
echo "  Time: $(date)" | tee -a "$LOG_FILE"
echo "  TTS: StyleTTS2-lite-vi" | tee -a "$LOG_FILE"
echo "  NS: DeepFilterNet" | tee -a "$LOG_FILE"
echo "  HF_HOME: $HF_HOME" | tee -a "$LOG_FILE"
echo "  PIP_CACHE_DIR: $PIP_CACHE_DIR" | tee -a "$LOG_FILE"
echo "==============================================" | tee -a "$LOG_FILE"

# --- Step 0: Ensure OS deps exist on fresh pod/container ---
# /workspace persists, but apt packages on container disk do not.
if ! command -v ffmpeg >/dev/null 2>&1 || ! dpkg -s libsndfile1 >/dev/null 2>&1 || ! command -v git >/dev/null 2>&1 || ! command -v espeak-ng >/dev/null 2>&1; then
    echo "[SETUP] Installing OS dependencies..." | tee -a "$LOG_FILE"
    export DEBIAN_FRONTEND=noninteractive
    stdbuf -oL apt-get update 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] apt-get update failed" | tee -a "$LOG_FILE"
        exit 1
    fi

    stdbuf -oL apt-get install -y --no-install-recommends \
        libsndfile1 ffmpeg git espeak-ng 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] apt-get install failed" | tee -a "$LOG_FILE"
        exit 1
    fi
fi

# --- Step 1: Create venv with system-site-packages ---
# RunPod image already has PyTorch + CUDA installed.
# --system-site-packages inherits them → avoids re-downloading ~2GB.
# VENV_VERSION: bump this to force venv recreation (e.g. after changing flags).
VENV_VERSION="v2-system-site-packages"
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

# --- Step 2: Install base deps (lightweight, NOT torch) ---
CURRENT_HASH=$(cat "${BASE_REQ_MARKER_INPUTS[@]}" | md5sum | cut -d' ' -f1 || echo "none")
SAVED_HASH=$(cat "$REQ_HASH_FILE" 2>/dev/null || echo "")

if [ "$CURRENT_HASH" != "$SAVED_HASH" ]; then
    echo "[SETUP] Installing base dependencies + build toolchain..." | tee -a "$LOG_FILE"

    stdbuf -oL python -m pip install --cache-dir "$PIP_CACHE_DIR" -c "$CONSTRAINTS_FILE" \
        "numpy>=2.0,<3" setuptools wheel packaging \
        flask flask-cors flask-sock faster-whisper soundfile huggingface_hub openai scipy 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] pip install (base) failed" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "$CURRENT_HASH" > "$REQ_HASH_FILE"
    echo "[SETUP] Base dependencies installed." | tee -a "$LOG_FILE"
else
    echo "[SETUP] Base dependencies already installed (cached)." | tee -a "$LOG_FILE"
fi

# --- Step 2b: Install StyleTTS2 TTS engine ---
if [ ! -f "$STYLETTS2_MARKER" ]; then
    echo "[SETUP] Installing StyleTTS2 + phonemizer..." | tee -a "$LOG_FILE"

    stdbuf -oL python -m pip install --cache-dir "$PIP_CACHE_DIR" -c "$CONSTRAINTS_FILE" \
        styletts2 phonemizer 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] pip install (styletts2) failed" | tee -a "$LOG_FILE"
        exit 1
    fi

    echo "$(date)" > "$STYLETTS2_MARKER"
    echo "[SETUP] StyleTTS2 installed." | tee -a "$LOG_FILE"
else
    echo "[SETUP] StyleTTS2 already installed (cached)." | tee -a "$LOG_FILE"
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

# --- Step 2d: Download StyleTTS2-lite-vi model from HuggingFace ---
STYLETTS2_DIR="${STYLETTS2_MODEL_DIR:-/workspace/styletts2_lite_vi}"
if [ ! -d "$STYLETTS2_DIR" ] || [ -z "$(ls -A "$STYLETTS2_DIR" 2>/dev/null)" ]; then
    echo "[SETUP] Downloading StyleTTS2-lite-vi model..." | tee -a "$LOG_FILE"
    stdbuf -oL python -c "
from huggingface_hub import snapshot_download
snapshot_download('dangtr0408/StyleTTS2-lite-vi', local_dir='$STYLETTS2_DIR')
print('Model downloaded to $STYLETTS2_DIR')
" 2>&1 | tee -a "$LOG_FILE"
    if [ ${PIPESTATUS[0]} -ne 0 ]; then
        echo "[ERROR] StyleTTS2 model download failed" | tee -a "$LOG_FILE"
        exit 1
    fi
else
    echo "[SETUP] StyleTTS2 model already downloaded." | tee -a "$LOG_FILE"
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
    from styletts2 import tts
    print("styletts2 OK")
except ImportError:
    print("styletts2 NOT installed — will try local inference.py")

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

# --- Step 4: Copy models to local SSD for fast loading ---
# Network volume (NFS) I/O is 5-10x slower than local SSD.
# Copy HF cache to /tmp (local SSD) → models load in ~20s instead of ~150s.
LOCAL_HF="/tmp/huggingface"
if [ ! -d "$LOCAL_HF/hub" ]; then
    echo "[SPEED] Copying models from NFS → local SSD..." | tee -a "$LOG_FILE"
    t_copy_start=$(date +%s)
    mkdir -p "$LOCAL_HF"
    cp -r "$HF_HOME/hub" "$LOCAL_HF/hub" 2>&1 | tee -a "$LOG_FILE"
    if [ -d "$HF_HOME/modules" ]; then
        cp -r "$HF_HOME/modules" "$LOCAL_HF/modules" 2>&1 | tee -a "$LOG_FILE"
    fi
    t_copy_end=$(date +%s)
    echo "[SPEED] Models copied in $((t_copy_end - t_copy_start))s" | tee -a "$LOG_FILE"
else
    echo "[SPEED] Models already on local SSD." | tee -a "$LOG_FILE"
fi
export HF_HOME="$LOCAL_HF"

# --- Step 5: Start server ---
echo "[START] Launching all_in_one_server.py on port $PORT..." | tee -a "$LOG_FILE"
stdbuf -oL python -u all_in_one_server.py 2>&1 | tee -a "$LOG_FILE"

# If we get here, server exited
EXIT_CODE=$?
echo "[EXIT] Server exited at $(date) with code $EXIT_CODE" | tee -a "$LOG_FILE"
