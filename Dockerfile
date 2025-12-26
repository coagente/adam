# Adam Trainer - Docker image for training on RunPod
# Built by coagente
#
# This image is based on RunPod's pytorch image which includes:
# - SSH access (via RunPod's system)
# - CUDA drivers
# - PyTorch pre-installed

FROM runpod/pytorch:2.1.0-py3.10-cuda11.8.0-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/coagente/adam"
LABEL org.opencontainers.image.description="Adam - Autonomous LLM trainer by coagente"
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /workspace/adam

# Install Python dependencies first (for better caching)
RUN pip install --no-cache-dir \
    typer \
    rich \
    pyyaml \
    runpod \
    paramiko \
    tomli \
    tomli-w \
    requests \
    tqdm \
    numpy \
    'transformers>=4.50' \
    accelerate \
    datasets \
    wandb \
    tiktoken \
    regex

# Copy project files
COPY elchat/ ./elchat/
COPY scripts/ ./scripts/
COPY tasks/ ./tasks/
COPY data/ ./data/
COPY configs/ ./configs/

# Add project to PYTHONPATH
ENV PYTHONPATH=/workspace/adam:$PYTHONPATH

# Environment variables for training
ENV PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_MODE=disabled

# Create startup script that runs on container start
# This script downloads data and starts training
RUN echo '#!/bin/bash\n\
set -e\n\
echo "========================================"\n\
echo "  Adam Trainer - by coagente"\n\
echo "========================================"\n\
echo ""\n\
\n\
# Default values from environment\n\
BASE_MODEL=${BASE_MODEL:-"LiquidAI/LFM2-2.6B-Exp"}\n\
TARGET_TOKENS=${TARGET_TOKENS:-50000000}\n\
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-4}\n\
NUM_SHARDS=${NUM_SHARDS:-3}\n\
\n\
echo "Configuration:"\n\
echo "  Base Model: $BASE_MODEL"\n\
echo "  Target Tokens: $TARGET_TOKENS"\n\
echo "  Batch Size: $DEVICE_BATCH_SIZE"\n\
echo "  Data Shards: $NUM_SHARDS"\n\
echo ""\n\
\n\
cd /workspace/adam\n\
\n\
# Download training data\n\
echo "Downloading training data..."\n\
python -m elchat.dataset -n $NUM_SHARDS\n\
\n\
# Run training\n\
echo "Starting training..."\n\
python -m scripts.cpt_train_spanish \\\n\
    --base_model=$BASE_MODEL \\\n\
    --target_tokens=$TARGET_TOKENS \\\n\
    --device_batch_size=$DEVICE_BATCH_SIZE\n\
\n\
echo ""\n\
echo "Training complete!"\n\
' > /start_training.sh && chmod +x /start_training.sh

# Don't use ENTRYPOINT - RunPod will keep the container running
# The user can run /start_training.sh manually or via SSH
CMD ["sleep", "infinity"]
