# Adam Trainer - Docker image for training on RunPod
# Built by coagente

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

LABEL org.opencontainers.image.source="https://github.com/coagente/adam"
LABEL org.opencontainers.image.description="Adam - Autonomous LLM trainer by coagente"

WORKDIR /workspace/adam

# Install additional dependencies
RUN pip install --no-cache-dir \
    typer rich pyyaml runpod paramiko tomli tomli-w \
    requests tqdm numpy tiktoken regex \
    'transformers>=4.50' accelerate datasets wandb

# Copy project
COPY elchat/ ./elchat/
COPY scripts/ ./scripts/
COPY tasks/ ./tasks/
COPY data/ ./data/
COPY configs/ ./configs/

ENV PYTHONPATH=/workspace/adam:$PYTHONPATH
ENV HF_HOME=/workspace/.cache/huggingface
ENV WANDB_MODE=disabled

# Create training script
RUN echo '#!/bin/bash\n\
cd /workspace/adam\n\
echo "Adam Trainer - by coagente"\n\
BASE_MODEL=${BASE_MODEL:-"LiquidAI/LFM2-2.6B-Exp"}\n\
TARGET_TOKENS=${TARGET_TOKENS:-50000000}\n\
BATCH=${DEVICE_BATCH_SIZE:-4}\n\
SHARDS=${NUM_SHARDS:-3}\n\
echo "Downloading data..."\n\
python -m elchat.dataset -n $SHARDS\n\
echo "Starting training..."\n\
python -m scripts.cpt_train_spanish --base_model=$BASE_MODEL --target_tokens=$TARGET_TOKENS --device_batch_size=$BATCH\n\
' > /start.sh && chmod +x /start.sh
