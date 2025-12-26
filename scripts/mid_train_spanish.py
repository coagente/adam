"""
Midtraining para elchat en ESPAÑOL.

Este script realiza el midtraining del modelo CPT para enseñarle:
- Formato de conversación (user/assistant)
- Tokens especiales de elchat
- Identidad en español
- Tareas básicas (matemáticas, ortografía, etc.)

Uso:
    python -m scripts.mid_train_spanish
    
    # Multi-GPU
    torchrun --standalone --nproc_per_node=8 -m scripts.mid_train_spanish

La diferencia principal con mid_train.py es que usa datos de identidad
en español y mezcla datasets en español con algunos en inglés.
"""

from collections import deque
import os
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
import time
import wandb
import torch
from contextlib import nullcontext
import torch.distributed as dist

from elchat.common import compute_init, compute_cleanup, print0, DummyWandb, autodetect_device_type

# -----------------------------------------------------------------------------
# Configuration
run = "dummy"
device_type = ""
model_tag = None
step = None
dtype = "bfloat16"
num_iterations = -1  # -1 = one epoch over data
max_seq_len = 2048
device_batch_size = 16
unembedding_lr = 0.0004  # Lower for CPT model
embedding_lr = 0.02
matrix_lr = 0.002
init_lr_frac = 1.0
weight_decay = 0.01
eval_every = 100
eval_tokens = 10 * 131072
total_batch_size = 131072
dry_run = 0

config_keys = [k for k, v in globals().items() if not k.startswith('_') and isinstance(v, (int, float, bool, str))]
exec(open(os.path.join('elchat', 'configurator.py')).read())
user_config = {k: globals()[k] for k in config_keys}

# -----------------------------------------------------------------------------
# Helper functions
def get_base_dir():
    if os.environ.get("ELCHAT_BASE_DIR"):
        return os.environ.get("ELCHAT_BASE_DIR")
    home_dir = os.path.expanduser("~")
    return os.path.join(home_dir, ".cache", "elchat")

# -----------------------------------------------------------------------------
# Compute init
device_type = autodetect_device_type() if device_type == "" else device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0
autocast_ctx = torch.amp.autocast(device_type=device_type, dtype=torch.bfloat16) if device_type == "cuda" else nullcontext()
synchronize = torch.cuda.synchronize if device_type == "cuda" else lambda: None
get_max_memory = torch.cuda.max_memory_allocated if device_type == "cuda" else lambda: 0

# wandb
use_dummy_wandb = run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="elchat-mid-es", name=run, config=user_config)

# -----------------------------------------------------------------------------
# Load model from CPT checkpoint
print0("Loading model from CPT checkpoint...")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer
except ImportError:
    raise ImportError("Please install transformers: pip install transformers")

base_dir = get_base_dir()

# Try to load from CPT checkpoint, fallback to base Qwen
cpt_checkpoint_dir = os.path.join(base_dir, "cpt_checkpoints", "qwen_cpt_es")
if os.path.exists(cpt_checkpoint_dir):
    # Find latest checkpoint
    import glob
    checkpoints = glob.glob(os.path.join(cpt_checkpoint_dir, "model_*.pt"))
    if checkpoints:
        latest_checkpoint = max(checkpoints, key=lambda x: int(x.split("_")[-1].split(".")[0]))
        print0(f"Loading CPT model from: {latest_checkpoint}")
        
        # Load base model structure
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        # Load CPT weights
        model.load_state_dict(torch.load(latest_checkpoint, map_location="cpu"))
    else:
        print0("No CPT checkpoint found, loading base Qwen2.5-0.5B")
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen2.5-0.5B",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
else:
    print0("CPT directory not found, loading base Qwen2.5-0.5B")
    model = AutoModelForCausalLM.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        torch_dtype=torch.bfloat16,
        trust_remote_code=True,
    )

tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B", trust_remote_code=True)
model = model.to(device)

# Compile for efficiency
orig_model = model
model = torch.compile(model, dynamic=False)

depth = model.config.num_hidden_layers
num_flops_per_token = 6 * sum(p.numel() for p in model.parameters())  # Approximate

tokens_per_fwdbwd = device_batch_size * max_seq_len
world_tokens_per_fwdbwd = tokens_per_fwdbwd * ddp_world_size
assert total_batch_size % world_tokens_per_fwdbwd == 0
grad_accum_steps = total_batch_size // world_tokens_per_fwdbwd

print0(f"Tokens / micro-batch / rank: {device_batch_size} x {max_seq_len} = {tokens_per_fwdbwd:,}")
print0(f"Total batch size {total_batch_size:,} => gradient accumulation steps: {grad_accum_steps}")

# -----------------------------------------------------------------------------
# Optimizer
optimizer = torch.optim.AdamW(
    [
        {"params": model.model.embed_tokens.parameters(), "lr": embedding_lr},
        {"params": model.lm_head.parameters(), "lr": unembedding_lr},
        {"params": [p for n, p in model.named_parameters() 
                    if "embed_tokens" not in n and "lm_head" not in n], "lr": matrix_lr},
    ],
    betas=(0.9, 0.95),
    eps=1e-8,
    weight_decay=weight_decay,
    fused=device_type == "cuda",
)

for group in optimizer.param_groups:
    group["lr"] = group["lr"] * init_lr_frac
    group["initial_lr"] = group["lr"]

# -----------------------------------------------------------------------------
# Load midtraining data
import json

def load_jsonl_conversations(filepath):
    """Load conversations from JSONL file."""
    conversations = []
    if not os.path.exists(filepath):
        print0(f"Warning: {filepath} not found")
        return conversations
    
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                data = json.loads(line)
                if "messages" in data:
                    conversations.append(data)
                elif isinstance(data, list):
                    # Handle format where line is just the messages array
                    conversations.append({"messages": data})
            except json.JSONDecodeError:
                continue
    
    print0(f"Loaded {len(conversations)} conversations from {filepath}")
    return conversations

def render_conversation_qwen(conversation, tokenizer, max_tokens=2048):
    """
    Render a conversation to token IDs using Qwen tokenizer.
    Returns (ids, mask) where mask=1 for assistant tokens to train on.
    """
    messages = conversation.get("messages", [])
    if not messages:
        return [], []
    
    ids = []
    mask = []
    
    # Add BOS if tokenizer has one
    if tokenizer.bos_token_id is not None:
        ids.append(tokenizer.bos_token_id)
        mask.append(0)
    
    for i, msg in enumerate(messages):
        role = msg.get("role", "")
        content = msg.get("content", "")
        
        if role == "system":
            # Merge system into first user message or skip
            continue
        
        # Format message
        if role == "user":
            text = f"<|im_start|>user\n{content}<|im_end|>\n"
            msg_ids = tokenizer.encode(text, add_special_tokens=False)
            ids.extend(msg_ids)
            mask.extend([0] * len(msg_ids))
        elif role == "assistant":
            text = f"<|im_start|>assistant\n{content}<|im_end|>\n"
            msg_ids = tokenizer.encode(text, add_special_tokens=False)
            ids.extend(msg_ids)
            # Train on assistant responses
            mask.extend([1] * len(msg_ids))
    
    # Truncate
    ids = ids[:max_tokens]
    mask = mask[:max_tokens]
    
    return ids, mask

# Load datasets
print0("Loading midtraining datasets...")

# Spanish identity conversations
identity_es_path = os.path.join(base_dir, "identity_conversations_es.jsonl")
if not os.path.exists(identity_es_path):
    # Try loading from data/ directory in repo
    identity_es_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "identity_conversations_es.jsonl")

identity_conversations = load_jsonl_conversations(identity_es_path)

# Try to load SmolTalk for general conversations
try:
    from datasets import load_dataset
    print0("Loading SmolTalk dataset...")
    smoltalk_ds = load_dataset("HuggingFaceTB/smol-smoltalk", split="train")
    smoltalk_conversations = [{"messages": row["messages"]} for row in smoltalk_ds.select(range(min(50000, len(smoltalk_ds))))]
    print0(f"Loaded {len(smoltalk_conversations)} SmolTalk conversations")
except Exception as e:
    print0(f"Could not load SmolTalk: {e}")
    smoltalk_conversations = []

# Combine all data
all_conversations = []

# Add identity conversations multiple times (they're important)
all_conversations.extend(identity_conversations * 5)

# Add SmolTalk
all_conversations.extend(smoltalk_conversations[:10000])  # Limit for speed

print0(f"Total midtraining conversations: {len(all_conversations)}")

# Shuffle
import random
random.seed(42)
random.shuffle(all_conversations)

# -----------------------------------------------------------------------------
# Data generator
last_step = False
approx_progress = 0.0

def mid_data_generator(split):
    global last_step, approx_progress
    
    dataset = all_conversations
    dataset_size = len(dataset)
    
    if dataset_size == 0:
        raise ValueError("No midtraining data loaded!")
    
    needed_tokens = device_batch_size * max_seq_len + 1
    token_buffer = deque()
    
    use_cuda = device.type == "cuda"
    scratch = torch.empty(needed_tokens, dtype=torch.int64, pin_memory=use_cuda)
    
    cursor = ddp_rank
    it = 0
    
    while True:
        while len(token_buffer) < needed_tokens:
            conversation = dataset[cursor]
            ids, _ = render_conversation_qwen(conversation, tokenizer, max_seq_len)
            
            if ids:
                token_buffer.extend(ids)
            
            cursor += ddp_world_size
            if cursor >= dataset_size:
                cursor -= dataset_size
                if split == "train":
                    last_step = True
        
        it += 1
        if 0 < num_iterations <= it and split == "train":
            last_step = True
        
        for i in range(needed_tokens):
            scratch[i] = token_buffer.popleft()
        
        inputs = scratch[:-1].view(device_batch_size, max_seq_len).to(device=device, non_blocking=use_cuda)
        targets = scratch[1:].view(device_batch_size, max_seq_len).to(device=device, non_blocking=use_cuda)
        
        if split == "train":
            if num_iterations > 0:
                approx_progress = it / num_iterations
            else:
                approx_progress = cursor / dataset_size
        
        yield inputs, targets

train_loader = mid_data_generator("train")
build_val_loader = lambda: mid_data_generator("val")

# LR scheduler
def get_lr_multiplier(progress):
    return 1 if progress < 0.8 else 1 - (progress - 0.8) / 0.2

# -----------------------------------------------------------------------------
# Training loop
x, y = next(train_loader)
min_val_loss = float("inf")
smooth_train_loss = 0
ema_beta = 0.9
total_training_time = 0
step = 0
progress = 0

print0("\nStarting midtraining...")
print0(f"  Dataset size: {len(all_conversations)}")
print0()

while True:
    flops_so_far = num_flops_per_token * total_batch_size * step
    
    # Sync last_step across ranks
    if ddp:
        last_step_tensor = torch.tensor(last_step, dtype=torch.int32, device=device)
        dist.all_reduce(last_step_tensor, op=dist.ReduceOp.MAX)
        last_step = bool(last_step_tensor.item())
    
    # Evaluation
    if eval_every > 0 and (last_step or step % eval_every == 0):
        model.eval()
        val_loader = build_val_loader()
        eval_steps = eval_tokens // (device_batch_size * max_seq_len * ddp_world_size)
        
        losses = []
        with torch.no_grad():
            for _ in range(min(eval_steps, 50)):
                try:
                    vx, vy = next(val_loader)
                    with autocast_ctx:
                        outputs = model(vx, labels=vy)
                        losses.append(outputs.loss.item())
                except StopIteration:
                    break
        
        if losses:
            val_loss = sum(losses) / len(losses)
            print0(f"Step {step:05d} | Val loss: {val_loss:.4f}")
            if val_loss < min_val_loss:
                min_val_loss = val_loss
            wandb_run.log({"step": step, "val_loss": val_loss})
        
        model.train()
    
    # Save checkpoint at end
    if master_process and last_step and not dry_run:
        output_dirname = "qwen_mid_es"
        checkpoint_dir = os.path.join(base_dir, "mid_checkpoints", output_dirname)
        os.makedirs(checkpoint_dir, exist_ok=True)
        
        model_path = os.path.join(checkpoint_dir, f"model_{step:06d}.pt")
        torch.save(orig_model.state_dict(), model_path)
        
        import json as json_module
        meta_path = os.path.join(checkpoint_dir, f"meta_{step:06d}.json")
        with open(meta_path, "w") as f:
            json_module.dump({
                "step": step,
                "val_loss": val_loss if 'val_loss' in dir() else None,
                "user_config": user_config,
            }, f, indent=2)
        
        print0(f"Saved midtrain checkpoint to {checkpoint_dir}")
    
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
        progress = max(progress, approx_progress)
    
    # LR schedule
    lrm = get_lr_multiplier(progress)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    
    optimizer.step()
    optimizer.zero_grad(set_to_none=True)
    
    synchronize()
    t1 = time.time()
    dt = t1 - t0
    
    # Logging
    step += 1
    smooth_train_loss = ema_beta * smooth_train_loss + (1 - ema_beta) * train_loss.item()
    debiased_loss = smooth_train_loss / (1 - ema_beta ** step)
    pct_done = 100 * progress
    tok_per_sec = int(total_batch_size / dt)
    
    if step > 10:
        total_training_time += dt
    
    print0(f"step {step:05d} ({pct_done:.2f}%) | loss: {debiased_loss:.4f} | lrm: {lrm:.2f} | dt: {dt*1000:.0f}ms | tok/sec: {tok_per_sec:,}")
    
    if step % 10 == 0:
        wandb_run.log({
            "step": step,
            "train_loss": debiased_loss,
            "lrm": lrm,
            "tok_per_sec": tok_per_sec,
        })

# Summary
print0(f"\n{'='*60}")
print0(f"Midtraining Complete!")
print0(f"  Total time: {total_training_time/60:.2f} minutes")
print0(f"  Min val loss: {min_val_loss:.4f}")
print0(f"  Peak memory: {get_max_memory()/1024/1024:.0f} MB")
print0(f"{'='*60}")

wandb_run.finish()
compute_cleanup()

