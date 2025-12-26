# Adam Trainer - Docker image for training on RunPod
# Built by coagente
# 
# Usage:
#   docker build -t ghcr.io/coagente/adam:latest .
#   docker push ghcr.io/coagente/adam:latest
#
# On RunPod, the container will auto-start training with:
#   python -m scripts.cpt_train_spanish --base_model=$BASE_MODEL ...

FROM runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/coagente/adam"
LABEL org.opencontainers.image.description="Adam - Autonomous LLM trainer by coagente"
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /workspace/adam

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy project files
COPY pyproject.toml .
COPY elchat/ ./elchat/
COPY scripts/ ./scripts/
COPY tasks/ ./tasks/
COPY data/ ./data/
COPY configs/ ./configs/

# Install Python dependencies
RUN pip install --no-cache-dir -e . \
    && pip install --no-cache-dir \
    'transformers>=4.55' \
    accelerate \
    datasets \
    wandb

# Pre-download common base models (optional, speeds up first run)
# Uncomment to include in image (increases size but speeds up training start)
# RUN python -c "from transformers import AutoModelForCausalLM; AutoModelForCausalLM.from_pretrained('LiquidAI/LFM2-2.6B-Exp', trust_remote_code=True)"

# Environment variables for training
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_MODE=disabled

# Default command - can be overridden via RunPod env vars
# The actual training command is constructed from environment variables:
#   BASE_MODEL, TARGET_TOKENS, DEVICE_BATCH_SIZE, etc.
COPY <<EOF /workspace/adam/entrypoint.sh
#!/bin/bash
set -e

echo "========================================"
echo "  Adam Trainer - by coagente"
echo "========================================"
echo ""

# Default values
BASE_MODEL=\${BASE_MODEL:-"LiquidAI/LFM2-2.6B-Exp"}
TARGET_TOKENS=\${TARGET_TOKENS:-50000000}
DEVICE_BATCH_SIZE=\${DEVICE_BATCH_SIZE:-4}
NUM_SHARDS=\${NUM_SHARDS:-3}

echo "Configuration:"
echo "  Base Model: \$BASE_MODEL"
echo "  Target Tokens: \$TARGET_TOKENS"
echo "  Batch Size: \$DEVICE_BATCH_SIZE"
echo "  Data Shards: \$NUM_SHARDS"
echo ""

# Download training data
echo "Downloading training data..."
python -m elchat.dataset -n \$NUM_SHARDS

# Run training
echo "Starting training..."
python -m scripts.cpt_train_spanish \
    --base_model=\$BASE_MODEL \
    --target_tokens=\$TARGET_TOKENS \
    --device_batch_size=\$DEVICE_BATCH_SIZE

echo ""
echo "Training complete!"
EOF

RUN chmod +x /workspace/adam/entrypoint.sh

ENTRYPOINT ["/workspace/adam/entrypoint.sh"]

