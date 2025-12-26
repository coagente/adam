"""
RunPod Serverless Handler for LFM2 Training.

This handler receives training requests via HTTP and executes
the training with gradient checkpointing, periodic saves to HuggingFace Hub,
and structured logging for monitoring.

Usage:
    # Local testing
    python handler.py --test

    # In production, RunPod calls handler() automatically
"""

import os
import sys
import json
import time
import logging
from typing import Optional
from contextlib import nullcontext

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)s | %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)
logger = logging.getLogger(__name__)

# RunPod SDK
try:
    import runpod
except ImportError:
    runpod = None
    logger.warning("RunPod SDK not installed. Local testing mode.")

# ML imports (deferred to handler for faster cold start)
torch = None
transformers = None


def setup_environment():
    """Setup CUDA and memory optimizations."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def load_dependencies():
    """Lazy load heavy dependencies."""
    global torch, transformers
    
    if torch is None:
        import torch as _torch
        torch = _torch
        logger.info(f"PyTorch {torch.__version__} loaded")
        
    if transformers is None:
        import transformers as _transformers
        transformers = _transformers
        logger.info(f"Transformers {transformers.__version__} loaded")


def get_device():
    """Get the best available device."""
    load_dependencies()
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
        logger.info(f"Using CUDA: {torch.cuda.get_device_name()}")
        logger.info(f"VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        return device, "cuda"
    else:
        logger.warning("CUDA not available, using CPU")
        return torch.device("cpu"), "cpu"


def upload_checkpoint_to_hf(
    model,
    tokenizer,
    repo_id: str,
    step: int,
    hf_token: str,
    is_final: bool = False
):
    """Upload checkpoint to HuggingFace Hub."""
    try:
        from huggingface_hub import HfApi
        
        # Create subfolder for checkpoint
        subfolder = "final" if is_final else f"checkpoint-{step}"
        
        logger.info(f"Uploading checkpoint to hf://{repo_id}/{subfolder}")
        
        # Save locally first
        local_path = f"/tmp/checkpoint-{step}"
        model.save_pretrained(local_path)
        tokenizer.save_pretrained(local_path)
        
        # Upload to Hub
        api = HfApi(token=hf_token)
        api.upload_folder(
            folder_path=local_path,
            repo_id=repo_id,
            path_in_repo=subfolder,
            commit_message=f"Checkpoint at step {step}",
        )
        
        logger.info(f"Checkpoint uploaded: hf://{repo_id}/{subfolder}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to upload checkpoint: {e}")
        return False


def download_training_data(num_shards: int = 5):
    """Download Spanish training data."""
    logger.info(f"Downloading {num_shards} shards of Spanish data...")
    
    # Import dataset module
    sys.path.insert(0, "/workspace/adam")
    from elchat.dataset import download_fineweb2_es
    
    download_fineweb2_es(num_shards=num_shards)
    logger.info("Data download complete")


def train_model(
    base_model: str = "LiquidAI/LFM2-2.6B-Exp",
    target_tokens: int = 100_000_000,
    device_batch_size: int = 1,
    gradient_accumulation_steps: int = 32,
    checkpoint_every: int = 100,
    num_shards: int = 5,
    hf_repo_id: Optional[str] = None,
    hf_token: Optional[str] = None,
    resume_from_checkpoint: Optional[str] = None,
):
    """
    Main training function with gradient checkpointing and HF uploads.
    
    Args:
        base_model: HuggingFace model ID to fine-tune
        target_tokens: Total tokens to train on
        device_batch_size: Batch size per device (keep low for memory)
        gradient_accumulation_steps: Accumulate gradients for effective larger batch
        checkpoint_every: Save checkpoint every N steps
        num_shards: Number of data shards to download
        hf_repo_id: HuggingFace repo for checkpoints (e.g. "coagente/adam")
        hf_token: HuggingFace API token
        resume_from_checkpoint: Path or HF repo to resume from
    """
    setup_environment()
    load_dependencies()
    
    device, device_type = get_device()
    
    # =========================================================================
    # Load Model with Gradient Checkpointing
    # =========================================================================
    logger.info(f"Loading model: {base_model}")
    
    model = transformers.AutoModelForCausalLM.from_pretrained(
        base_model,
        torch_dtype=torch.bfloat16 if device_type == "cuda" else torch.float32,
        trust_remote_code=True,
        attn_implementation="eager",  # Avoid flash attention issues
    )
    tokenizer = transformers.AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)
    
    # Enable gradient checkpointing to reduce VRAM
    model.gradient_checkpointing_enable()
    logger.info("Gradient checkpointing enabled (-40% VRAM)")
    
    model = model.to(device)
    
    num_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Model loaded. Parameters: {num_params:,}")
    
    # =========================================================================
    # Training Configuration
    # =========================================================================
    max_seq_len = 2048
    total_batch_size = device_batch_size * gradient_accumulation_steps * max_seq_len
    num_iterations = target_tokens // total_batch_size
    
    logger.info(f"Training config:")
    logger.info(f"  Target tokens: {target_tokens:,}")
    logger.info(f"  Iterations: {num_iterations}")
    logger.info(f"  Batch size: {device_batch_size} x {gradient_accumulation_steps} accum")
    logger.info(f"  Effective batch: {total_batch_size:,} tokens/step")
    
    # =========================================================================
    # Download Data
    # =========================================================================
    download_training_data(num_shards)
    
    # =========================================================================
    # Setup Optimizer
    # =========================================================================
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5,  # Low LR for fine-tuning
        betas=(0.9, 0.95),
        weight_decay=0.01,
    )
    
    # =========================================================================
    # Data Loader
    # =========================================================================
    def data_loader():
        """Yield batches of tokenized Spanish text."""
        import pyarrow.parquet as pq
        from collections import deque
        
        base_dir = os.path.expanduser("~/.cache/elchat/base_data")
        parquet_files = sorted([
            os.path.join(base_dir, f) for f in os.listdir(base_dir)
            if f.endswith('.parquet')
        ])
        
        token_buffer = deque()
        needed = device_batch_size * max_seq_len + 1
        
        while True:
            for filepath in parquet_files:
                pf = pq.ParquetFile(filepath)
                for rg_idx in range(pf.num_row_groups):
                    rg = pf.read_row_group(rg_idx)
                    texts = rg.column('text').to_pylist()
                    
                    for text in texts:
                        if not text or len(text) < 100:
                            continue
                        ids = tokenizer.encode(text, add_special_tokens=False)
                        token_buffer.extend(ids)
                        
                        while len(token_buffer) >= needed:
                            batch = [token_buffer.popleft() for _ in range(needed)]
                            inputs = torch.tensor(batch[:-1]).view(device_batch_size, max_seq_len).to(device)
                            targets = torch.tensor(batch[1:]).view(device_batch_size, max_seq_len).to(device)
                            yield inputs, targets
    
    loader = data_loader()
    
    # =========================================================================
    # Training Loop
    # =========================================================================
    logger.info("Starting training...")
    
    model.train()
    step = 0
    total_loss = 0
    start_time = time.time()
    
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
    
    while step < num_iterations:
        optimizer.zero_grad()
        
        # Gradient accumulation
        for micro_step in range(gradient_accumulation_steps):
            inputs, targets = next(loader)
            
            with autocast_ctx:
                outputs = model(inputs, labels=targets)
                loss = outputs.loss / gradient_accumulation_steps
            
            loss.backward()
            total_loss += loss.item()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        
        optimizer.step()
        step += 1
        
        # Logging
        if step % 10 == 0:
            avg_loss = total_loss / 10
            elapsed = time.time() - start_time
            tokens_per_sec = (step * total_batch_size) / elapsed
            pct = 100 * step / num_iterations
            
            logger.info(
                f"Step {step:05d}/{num_iterations} ({pct:.1f}%) | "
                f"Loss: {avg_loss:.4f} | "
                f"Tok/s: {tokens_per_sec:,.0f}"
            )
            
            total_loss = 0
        
        # Checkpoint
        if checkpoint_every > 0 and step % checkpoint_every == 0:
            if hf_repo_id and hf_token:
                upload_checkpoint_to_hf(
                    model, tokenizer, hf_repo_id, step, hf_token
                )
            else:
                # Local save
                local_path = f"/workspace/checkpoints/step-{step}"
                os.makedirs(local_path, exist_ok=True)
                model.save_pretrained(local_path)
                tokenizer.save_pretrained(local_path)
                logger.info(f"Checkpoint saved: {local_path}")
    
    # =========================================================================
    # Final Save
    # =========================================================================
    logger.info("Training complete!")
    
    if hf_repo_id and hf_token:
        upload_checkpoint_to_hf(
            model, tokenizer, hf_repo_id, step, hf_token, is_final=True
        )
    
    total_time = time.time() - start_time
    logger.info(f"Total training time: {total_time/60:.1f} minutes")
    
    return {
        "status": "completed",
        "steps": step,
        "final_loss": avg_loss if 'avg_loss' in dir() else None,
        "training_time_minutes": total_time / 60,
        "model_repo": hf_repo_id,
    }


def handler(job):
    """
    RunPod Serverless handler.
    
    Expected job input:
    {
        "base_model": "LiquidAI/LFM2-2.6B-Exp",
        "target_tokens": 100000000,
        "device_batch_size": 1,
        "gradient_accumulation_steps": 32,
        "checkpoint_every": 100,
        "num_shards": 5,
        "hf_repo_id": "coagente/adam",
        "hf_token": "hf_xxx"
    }
    """
    job_input = job.get("input", {})
    
    logger.info("="*60)
    logger.info("Adam Training Job Started")
    logger.info("="*60)
    logger.info(f"Job ID: {job.get('id', 'unknown')}")
    logger.info(f"Input: {json.dumps(job_input, indent=2)}")
    
    try:
        result = train_model(
            base_model=job_input.get("base_model", "LiquidAI/LFM2-2.6B-Exp"),
            target_tokens=job_input.get("target_tokens", 100_000_000),
            device_batch_size=job_input.get("device_batch_size", 1),
            gradient_accumulation_steps=job_input.get("gradient_accumulation_steps", 32),
            checkpoint_every=job_input.get("checkpoint_every", 100),
            num_shards=job_input.get("num_shards", 5),
            hf_repo_id=job_input.get("hf_repo_id"),
            hf_token=job_input.get("hf_token"),
            resume_from_checkpoint=job_input.get("resume_from_checkpoint"),
        )
        
        logger.info("="*60)
        logger.info("Training Complete!")
        logger.info(f"Result: {json.dumps(result, indent=2)}")
        logger.info("="*60)
        
        return result
        
    except Exception as e:
        logger.error(f"Training failed: {e}")
        import traceback
        traceback.print_exc()
        return {"status": "error", "error": str(e)}


# RunPod Serverless entrypoint
if runpod:
    runpod.serverless.start({"handler": handler})
else:
    # Local testing
    if __name__ == "__main__":
        import argparse
        parser = argparse.ArgumentParser()
        parser.add_argument("--test", action="store_true", help="Run local test")
        args = parser.parse_args()
        
        if args.test:
            test_job = {
                "id": "test-local",
                "input": {
                    "base_model": "LiquidAI/LFM2-2.6B-Exp",
                    "target_tokens": 1_000_000,  # Small for testing
                    "device_batch_size": 1,
                    "gradient_accumulation_steps": 4,
                    "checkpoint_every": 0,  # Disable for test
                    "num_shards": 1,
                }
            }
            result = handler(test_job)
            print(json.dumps(result, indent=2))

