#!/bin/bash
# Setup Matcha-TTS server on RunPod pod
# Usage: bash /workspace/voicebot-pipeline/setup-matcha.sh

set -e

echo "=== Installing espeak-ng ==="
apt-get update -qq && apt-get install -y -qq espeak-ng 2>/dev/null || true

echo "=== Installing Matcha-TTS dependencies ==="
pip install -q flask flask-cors huggingface_hub soundfile numpy
pip install -q git+https://github.com/phineas-pta/MatchaTTS_ngngngan.git

echo ""
echo "=== Done! Start server with: ==="
echo "python /workspace/voicebot-pipeline/matcha_tts_server.py"
echo ""
echo "Environment variables (optional):"
echo "  MATCHA_STEPS=10    # denoising steps (10=fast, 50=quality)"
echo "  MATCHA_TEMP=0.667  # sampling temperature"
echo "  PORT=7860          # server port"
