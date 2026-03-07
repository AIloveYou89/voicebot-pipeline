#!/bin/bash
# Setup Claude Code on RunPod pod
# Usage: bash /workspace/voicebot-pipeline/setup-pod.sh
# Lưu trên Network Volume — persist qua restart

set -e

echo "=== Installing Node.js + Claude Code ==="
if ! command -v node &>/dev/null; then
    curl -fsSL https://nodejs.org/dist/v20.18.0/node-v20.18.0-linux-x64.tar.xz | tar -xJ -C /usr/local --strip-components=1
    echo "Node.js installed: $(node -v)"
fi

if ! command -v claude &>/dev/null; then
    npm install -g @anthropic-ai/claude-code
    echo "Claude Code installed"
fi

echo "=== Installing espeak-ng ==="
apt install -y espeak-ng 2>/dev/null || true

echo "=== Installing Flask ==="
pip install flask flask-cors 2>/dev/null || true

echo ""
echo "=== Done! Running Claude Code ==="
claude
