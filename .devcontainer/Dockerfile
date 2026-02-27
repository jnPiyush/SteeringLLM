# SteeringLLM Dev Container
# Plain Python 3.12 on Debian Bookworm -- NO Microsoft devcontainer base image.
# Using the official Python image avoids the mcr.microsoft.com/devcontainers/python
# bootstrapping script that auto-installs the .NET SDK.
FROM python:3.12-bookworm

# Create the non-root vscode user (mirrors what the MS devcontainer base does,
# but without pulling in .NET or other unneeded toolchains)
RUN groupadd --gid 1000 vscode \
    && useradd --uid 1000 --gid vscode --shell /bin/bash --create-home vscode

# System dependencies for PDF processing and general build tools
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    git \
    curl \
    sudo \
    && echo 'vscode ALL=(ALL) NOPASSWD:ALL' >> /etc/sudoers.d/vscode \
    && rm -rf /var/lib/apt/lists/*

# Install Node.js 20 LTS via NodeSource
RUN curl -fsSL https://deb.nodesource.com/setup_20.x | bash - \
    && apt-get install -y --no-install-recommends nodejs \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip

# Pre-install PyTorch CPU (avoids re-downloading on every rebuild).
# Swap the index URL to cu121 for GPU Codespaces.
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
USER vscode
