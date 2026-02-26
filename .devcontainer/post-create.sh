#!/usr/bin/env bash
# Post-create script for SteeringLLM dev container
# Runs after the container is built and the repo is cloned
set -euo pipefail

echo "=== SteeringLLM post-create setup ==="

# Create HF cache directory inside the /workspaces volume (persists across rebuilds)
mkdir -p /workspaces/.cache/huggingface

# Install the package in editable mode with demo + dev extras
pip install --no-cache-dir -e ".[demo,dev]"

# Pre-download the default model (microsoft/phi-2) so first demo launch is instant
echo "Pre-downloading microsoft/phi-2 model (this may take a few minutes)..."
python -c "
from transformers import AutoModelForCausalLM, AutoTokenizer
print('Downloading tokenizer...')
AutoTokenizer.from_pretrained('microsoft/phi-2')
print('Downloading model...')
AutoModelForCausalLM.from_pretrained('microsoft/phi-2')
print('Model cached successfully.')
" || echo "[WARN] Model pre-download failed -- will download on first demo launch."

echo "=== Setup complete ==="
echo "Launch the demo with: python demo/launch.py"
