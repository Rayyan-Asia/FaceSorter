#!/bin/bash
set -e
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

python3 -m venv venv
./venv/bin/pip install --upgrade pip

# Install platform-specific onnxruntime variant.
# These packages conflict with each other so we pick exactly one.
PLATFORM="$(uname -s)"
ARCH="$(uname -m)"

if [ "$PLATFORM" = "Darwin" ]; then
    echo "macOS ($ARCH) detected — installing onnxruntime with CoreML support"
    ./venv/bin/pip install onnxruntime
elif [ "$PLATFORM" = "Linux" ]; then
    if command -v nvidia-smi &>/dev/null 2>&1; then
        echo "NVIDIA GPU detected — installing onnxruntime-gpu"
        ./venv/bin/pip install onnxruntime-gpu
    else
        echo "No NVIDIA GPU — installing onnxruntime (CPU)"
        ./venv/bin/pip install onnxruntime
    fi
else
    echo "Unknown platform ($PLATFORM) — installing onnxruntime (CPU)"
    ./venv/bin/pip install onnxruntime
fi

./venv/bin/pip install -r requirements.txt
echo "Python environment ready."
