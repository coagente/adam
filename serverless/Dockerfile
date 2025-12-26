# Adam Training Worker for RunPod Serverless
#
# Build:
#   docker build -t ghcr.io/coagente/adam-worker:latest .
#
# Push:
#   docker push ghcr.io/coagente/adam-worker:latest
#
# Local test:
#   docker run --gpus all -it ghcr.io/coagente/adam-worker:latest python handler.py --test

FROM runpod/pytorch:2.2.1-py3.10-cuda12.1.1-devel-ubuntu22.04

WORKDIR /workspace

# Install Python dependencies
RUN pip install --no-cache-dir \
    runpod \
    transformers>=4.50 \
    accelerate \
    datasets \
    huggingface_hub \
    pyarrow \
    tiktoken \
    regex \
    requests \
    tqdm \
    wandb

# Copy handler
COPY handler.py /workspace/handler.py

# Clone the adam repo for the dataset module
RUN git clone https://github.com/coagente/adam.git /workspace/adam

# Set Python path
ENV PYTHONPATH=/workspace/adam:$PYTHONPATH

# RunPod Serverless expects the handler to be the entrypoint
CMD ["python", "/workspace/handler.py"]

