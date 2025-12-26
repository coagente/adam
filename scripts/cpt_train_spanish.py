"""
Continual Pre-training (CPT) from Qwen2.5-0.5B for Spanish elchat.

This script loads a pre-trained Qwen2.5-0.5B model and continues training
on Spanish data. This is much more efficient than training from scratch
because the model already has strong language understanding capabilities.

Usage:
    # Single GPU
    python -m scripts.cpt_train_spanish
    
    # Multi-GPU with torchrun
    torchrun --standalone --nproc_per_node=8 -m scripts.cpt_train_spanish

Key differences from base_train.py:
1. Loads pre-trained Qwen2.5-0.5B instead of random initialization
2. Uses lower learning rates (model is already trained)
3. Needs fewer tokens (~1-2B vs ~11B for from-scratch)
4. Uses Spanish data from FineWeb-2 or CulturaX
"""

import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
from contextlib import nullcontext

import wandb
import torch
import torch.nn.functional as F

from elchat.common import compute_init, compute_cleanup, print0, DummyWandb, print_banner, autodetect_device_type

print_banner()

# -----------------------------------------------------------------------------
# User settings
run = "dummy"  # wandb run name
device_type = ""  # cuda|cpu|mps (empty => autodetect)

# Model to load
base_model = "Qwen/Qwen2.5-0.5B"  # Pre-trained model to continue from

# Training horizon - much smaller than from-scratch training
num_iterations = -1  # explicit number of steps (-1 = disable, use target_tokens)
target_tokens = 1_500_000_000  # 1.5B tokens for CPT (vs ~11B for from-scratch)

# Optimization - lower learning rates for CPT
device_batch_size = 16  # per-device batch size
total_batch_size = 131072  # ~128K tokens per step (smaller than original 512K)
max_seq_len = 2048  # context length

# Learning rates - much lower for CPT to avoid catastrophic forgetting
embedding_lr = 0.02  # 10x lower than from-scratch (0.2)
unembedding_lr = 0.0004  # 10x lower than from-scratch (0.004)
matrix_lr = 0.002  # 10x lower than from-scratch (0.02)
weight_decay = 0.01  # Small weight decay helps prevent forgetting
grad_clip = 1.0

# LR schedule
warmup_ratio = 0.05  # 5% warmup
warmdown_ratio = 0.1  # 10% warmdown
final_lr_frac = 0.1  # End at 10% of peak LR

# Evaluation
eval_every = 100
eval_tokens = 10 * 131072
sample_every = 500

# Output
model_tag = "qwen_cpt_es"

# CLI overrides
config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('elchat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# Autocast setup - MPS supports float16 but not bfloat16
if device_type == "cuda":
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16)
    synchronize = torch.cuda.synchronize
    get_max_memory = torch.cuda.max_memory_allocated
elif device_type == "mps":
    autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.float16)
    synchronize = lambda: torch.mps.synchronize()
    get_max_memory = lambda: torch.mps.current_allocated_memory()
else:
    autocast_ctx = nullcontext()
    synchronize = lambda: None
    get_max_memory = lambda: 0

# wandb
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="elchat-cpt-es", name=run, config=user_config)

# -----------------------------------------------------------------------------
# Load pre-trained model
print0(f"Loading pre-trained model: {base_model}")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

# Load Qwen2.5-0.5B
# Use float16 for MPS, bfloat16 for CUDA
model_dtype = torch.float16 if device_type == "mps" else torch.bfloat16
qwen_model = AutoModelForCausalLM.from_pretrained(
    base_model,
    torch_dtype=model_dtype,
    trust_remote_code=True,
)
qwen_tokenizer = AutoTokenizer.from_pretrained(base_model, trust_remote_code=True)

# Move to device
qwen_model = qwen_model.to(device)
print0(f"Model loaded. Parameters: {sum(p.numel() for p in qwen_model.parameters()):,}")

# Get model config
model_config = qwen_model.config
vocab_size = model_config.vocab_size
num_layers = model_config.num_hidden_layers
hidden_size = model_config.hidden_size
num_heads = model_config.num_attention_heads
num_kv_heads = getattr(model_config, 'num_key_value_heads', num_heads)

print0(f"Model config:")
print0(f"  vocab_size: {vocab_size}")
print0(f"  num_layers: {num_layers}")
print0(f"  hidden_size: {hidden_size}")
print0(f"  num_heads: {num_heads}")
print0(f"  num_kv_heads: {num_kv_heads}")

# Compile model for efficiency (skip on MPS - not well supported)
# NOTE: torch.compile disabled due to FlashAttention dtype issues on some GPUs
# if device_type == "cuda":
#     model = torch.compile(qwen_model, dynamic=False)
# else:
#     model = qwen_model  # Skip compilation on MPS/CPU
model = qwen_model  # No compilation for now

# Estimate FLOPs
def estimate_flops(model):
    nparams = sum(p.numel() for p in model.parameters())
    # Try to get embedding size (different models have different structures)
    try:
        if hasattr(model, 'model') and hasattr(model.model, 'embed_tokens'):
            nparams_embedding = model.model.embed_tokens.weight.numel()
        elif hasattr(model, 'transformer') and hasattr(model.transformer, 'wte'):
            nparams_embedding = model.transformer.wte.weight.numel()
        else:
            nparams_embedding = vocab_size * hidden_size  # Estimate
    except:
        nparams_embedding = vocab_size * hidden_size
    l = num_layers
    h = num_heads
    q = hidden_size // num_heads
    t = max_seq_len
    num_flops_per_token = 6 * (nparams - nparams_embedding) + 12 * l * h * q * t
    return num_flops_per_token

num_flops_per_token = estimate_flops(qwen_model)
print0(f"Estimated FLOPs per token: {num_flops_per_token:e}")

# Calculate number of iterations
tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

if num_iterations > 0:
    print0(f"Using explicit num_iterations: {num_iterations}")
else:
    num_iterations = target_tokens // total_batch_size
    print0(f"Calculated num_iterations from target_tokens: {num_iterations}")

total_tokens = total_batch_size * num_iterations
print0(f"Total batch size: {total_batch_size:,} => grad_accum_steps: {grad_accum_steps}")
print0(f"Total training tokens: {total_tokens:,}")

# -----------------------------------------------------------------------------
# Optimizer setup
# We use AdamW for all parameters (simpler than Muon for CPT)
# Separate parameters into groups by name to avoid duplicates
embed_params = []
lm_head_params = []
other_params = []

for name, param in qwen_model.named_parameters():
    if not param.requires_grad:
        continue
    if "embed_tokens" in name:
        embed_params.append(param)
    elif "lm_head" in name:
        lm_head_params.append(param)
    else:
        other_params.append(param)

param_groups = [
    {"params": embed_params, "lr": embedding_lr},
    {"params": lm_head_params, "lr": unembedding_lr},
    {"params": other_params, "lr": matrix_lr},
]

optimizer = torch.optim.AdamW(
    param_groups,
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=weight_decay,
    fused=device_type == "cuda",
)

# Save initial LRs
for group in optimizer.param_groups:
    group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Data loader for Spanish data
def get_base_dir():
    if os.environ.get("ELCHAT_BASE_DIR"):
        return os.environ.get("ELCHAT_BASE_DIR")
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, ".cache", "elchat")

def spanish_data_loader(batch_size, seq_len, device):
    """Load Spanish data and yield batches of token ids."""
    import pyarrow.parquet as pq
    from collections import deque
    
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "base_data")
    
    parquet_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    
    if not parquet_files:
        raise FileNotFoundError(f"No data found in {data_dir}. Run download_spanish_data.py first.")
    
    print0(f"Loading Spanish data from {len(parquet_files)} parquet files")
    
    # Use Qwen tokenizer
    bos_token_id = qwen_tokenizer.bos_token_id or 0
    needed_tokens = batch_size * seq_len + 1
    token_buffer = deque()
    
    use_cuda = device.type == "cuda"
    scratch = torch.empty(needed_tokens, dtype=torch.long, pin_memory=use_cuda)
    
    pq_idx = ddp_rank  # Start at rank offset for distributed training
    while True:
        for filepath in parquet_files:
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pq_idx % pf.num_row_groups, pf.num_row_groups, ddp_world_size):
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                
                for text in texts:
                    if not text or len(text) < 100:
                        continue
                    
                    # Tokenize
                    ids = qwen_tokenizer.encode(text, add_special_tokens=False)
                    token_buffer.append(bos_token_id)
                    token_buffer.extend(ids)
                    
                    # Yield batches when we have enough
                    while len(token_buffer) >= needed_tokens:
                        for i in range(needed_tokens):
                            scratch[i] = token_buffer.popleft()
                        
                        inputs = scratch[:-1].view(batch_size, seq_len).to(device=device, non_blocking=use_cuda)
                        targets = scratch[1:].view(batch_size, seq_len).to(device=device, non_blocking=use_cuda)
                        yield inputs, targets
        
        pq_idx = ddp_rank  # Reset for next epoch

train_loader = spanish_data_loader(device_batch_size, max_seq_len, device)
x, y = next(train_loader)

# -----------------------------------------------------------------------------
# LR scheduler
def get_lr_multiplier(it):
    warmup_iters = round(warmup_ratio * num_iterations)
    warmdown_iters = round(warmdown_ratio * num_iterations)
    if it < warmup_iters:
        return (it + 1) / warmup_iters
    elif it <= num_iterations - warmdown_iters:
        return 1.0
    else:
        progress = (num_iterations - it) / warmdown_iters
        return progress * 1.0 + (1 - progress) * final_lr_frac

# -----------------------------------------------------------------------------
# Training loop
step = 0
min_val_loss = float("inf")
smooth_train_loss = 0
total_training_time = 0

print0(f"\nStarting CPT training...")
print0(f"  Model: {base_model}")
print0(f"  Target tokens: {total_tokens:,}")
print0(f"  Iterations: {num_iterations:,}")
print0()

while True:
    last_step = step == num_iterations
    
    # Evaluation
    if last_step or step % eval_every == 0:
        model.eval()
        eval_losses = []
        eval_loader = spanish_data_loader(device_batch_size, max_seq_len, device)
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        
        with torch.no_grad():
            for _ in range(eval_steps):
                ex, ey = next(eval_loader)
                with autocast_ctx:
                    outputs = model(ex, labels=ey)
                    eval_losses.append(outputs.loss.item())
        
        val_loss = sum(eval_losses) / len(eval_losses)
        print0(f"Step {step:05d} | Val loss: {val_loss:.4f}")
        
        if val_loss < min_val_loss:
            min_val_loss = val_loss
        
        wandb_run.log({"step": step, "val_loss": val_loss})
        model.train()
    
    # Sampling
    if master_process and (last_step or (step > 0 and step % sample_every == 0)):
        model.eval()
        prompts = [
            "La capital de España es",
            "El símbolo químico del oro es",
            "Los planetas del sistema solar son:",
            "Si ayer fue viernes, entonces mañana será",
        ]
        
        print0("\nSamples:")
        for prompt in prompts:
            input_ids = qwen_tokenizer.encode(prompt, return_tensors="pt").to(device)
            with torch.no_grad(), autocast_ctx:
                output_ids = model.generate(
                    input_ids,
                    max_new_tokens=32,
                    do_sample=False,
                    pad_token_id=qwen_tokenizer.eos_token_id,
                )
            output_text = qwen_tokenizer.decode(output_ids[0], skip_special_tokens=True)
            print0(f"  {output_text}")
        print0()
        model.train()
    
    # Save checkpoint at end
    if master_process and last_step:
        base_dir = get_base_dir()
        checkpoint_dir = os.path.join(base_dir, "cpt_checkpoints", model_tag)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        # Save model
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(qwen_model.state_dict(), model_path)
        
        # Save metadata
        import json
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w") as f:
            json.dump({
                "step": step,
                "val_loss": val_loss,
                "base_model": base_model,
                "total_tokens": total_tokens,
                "user_config": user_config,
            }, f, indent=2)
        
        print0(f"Saved checkpoint to {checkpoint_dir}")
    
    if last_step:
        break
    
    # Training step
    synchronize()
    t0 = time.time()
    
    for micro_step in range(grad_accum_steps):
        with autocast_ctx:
            outputs = model(x, labels=y)
            loss = outputs.loss
        
        train_loss = loss.detach()
        loss = loss / grad_accum_steps
        loss.backward()
        x, y = next(train_loader)
    
    # Gradient clipping
    if grad_clip > 0.0:
        grad_norm = torch.nn.utils.clip_grad_norm_(qwen_model.parameters(), grad_clip)
    
    # LR schedule
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    # Logging
    ema_beta = 0.9
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_loss = smooth_train_loss / (1 - ema_beta ** (step + 1))
    
    pct_done = 100 * step / num_iterations
    tok_per_sec = int(total_batch_size / dt)
    
    if step > 10:
        total_training_time += dt
    
    print0(f"step {step:05d}/{num_iterations:05d} ({pct_done:.2f}%) | loss: {debiased_loss:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,}")
    
    if step % 50 == 0:
        wandb_run.log({
            "step": step,
            "train_loss": debiased_loss,
            "lrm": lrm,
            "tok_per_sec": tok_per_sec,
        })
    
    step += 1

# Summary
print0(f"\n{'='*60}")
print0(f"CPT Training Complete!")
print0(f"  Total training time: {total_training_time/60:.2f} minutes")
print0(f"  Minimum val loss: {min_val_loss:.4f}")
print0(f"  Peak memory: {get_max_memory()/1024/1024:.0f} MB")
print0(f"{'='*60}")

wandb_run.finish()
compute_cleanup()

