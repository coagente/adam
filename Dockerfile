# Adam Trainer - Docker image for training on RunPod
# Built by coagente
#
# Uses RunPod's slim base image with CUDA support

FROM runpod/base:0.6.2-cuda12.2.0

LABEL org.opencontainers.image.source="https://github.com/coagente/adam"
LABEL org.opencontainers.image.description="Adam - Autonomous LLM trainer by coagente"
LABEL org.opencontainers.image.licenses="MIT"

# Set working directory
WORKDIR /workspace/adam

# Install PyTorch and dependencies
RUN pip install --no-cache-dir \
    torch \
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

# Create startup script
RUN echo '#!/bin/bash\n\
set -e\n\
echo "========================================"\n\
echo "  Adam Trainer - by coagente"\n\
echo "========================================"\n\
BASE_MODEL=${BASE_MODEL:-"LiquidAI/LFM2-2.6B-Exp"}\n\
TARGET_TOKENS=${TARGET_TOKENS:-50000000}\n\
DEVICE_BATCH_SIZE=${DEVICE_BATCH_SIZE:-4}\n\
NUM_SHARDS=${NUM_SHARDS:-3}\n\
echo "Config: Model=$BASE_MODEL, Tokens=$TARGET_TOKENS, Batch=$DEVICE_BATCH_SIZE"\n\
cd /workspace/adam\n\
python -m elchat.dataset -n $NUM_SHARDS\n\
python -m scripts.cpt_train_spanish --base_model=$BASE_MODEL --target_tokens=$TARGET_TOKENS --device_batch_size=$DEVICE_BATCH_SIZE\n\
echo "Training complete!"\n\
' > /start_training.sh && chmod +x /start_training.sh

# RunPod will handle SSH and keep container alive
CMD ["sleep", "infinity"]
