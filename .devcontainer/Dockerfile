# SteeringLLM Dev Container
# Python 3.12 on Debian Bookworm -- lightweight base for ML workloads
FROM mcr.microsoft.com/devcontainers/python:1-3.12-bookworm

# System dependencies for PDF processing and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Pre-install PyTorch CPU (avoids re-downloading on every rebuild)
# CPU-only keeps the image smaller; swap index URL to cu121 for GPU Codespaces
RUN pip install --no-cache-dir \
    torch --index-url https://download.pytorch.org/whl/cpu

# Pre-install heavy ML dependencies so post-create is fast.
# Version specifiers MUST be quoted -- bare > is a shell redirect in RUN commands.
RUN pip install --no-cache-dir \
    "transformers>=4.36.0,<5.0.0" \
    "numpy>=1.24.0,<2.0.0" \
    "scikit-learn>=1.3.0,<2.0.0" \
    "streamlit>=1.30.0,<2.0.0" \
    "pandas>=2.0.0,<3.0.0" \
    "PyPDF2>=3.0.0,<4.0.0" \
    "pdfplumber>=0.10.0,<1.0.0"

# /workspaces is a Codespaces runtime volume mount -- not available at build time.
# Cache directory creation is handled in post-create.sh instead.

WORKDIR /workspaces/SteeringLLM
