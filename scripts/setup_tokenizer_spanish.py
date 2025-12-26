"""
Setup tokenizer for Spanish elchat.

Two options are available:
1. Use Qwen2.5 tokenizer (recommended for CPT approach)
2. Train a new BPE tokenizer on Spanish data

Usage:
    # Option 1: Use Qwen2.5 tokenizer
    python -m scripts.setup_tokenizer_spanish --source qwen2.5
    
    # Option 2: Train on Spanish data
    python -m scripts.setup_tokenizer_spanish --source train --vocab_size 65536

The tokenizer will be saved to ~/.cache/elchat/tokenizer/
"""

import os
import argparse
import pickle
import tiktoken

# Import elchat's special tokens and patterns
from elchat.tokenizer import SPECIAL_TOKENS, SPLIT_PATTERN

def get_base_dir():
    """Get the base directory for Spanish elchat data."""
    if os.environ.get("ELCHAT_BASE_DIR"):
        return os.environ.get("ELCHAT_BASE_DIR")
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    return os.path.join(cache_dir, "elchat")

def setup_qwen_tokenizer(output_dir: str):
    """
    Setup tokenizer based on Qwen2.5 vocabulary.
    Qwen2.5 has excellent multilingual support including Spanish.
    
    We save the HuggingFace tokenizer directly for compatibility.
    """
    print("Setting up Qwen2.5 tokenizer for Spanish...")
    
    try:
        from transformers import AutoTokenizer
    except ImportError:
        raise ImportError("Please install transformers: pip install transformers")
    
    # Load Qwen2.5 tokenizer
    print("Loading Qwen/Qwen2.5-0.5B tokenizer...")
    qwen_tokenizer = AutoTokenizer.from_pretrained(
        "Qwen/Qwen2.5-0.5B",
        trust_remote_code=True,
    )
    
    # Get vocabulary info
    vocab = qwen_tokenizer.get_vocab()
    vocab_size = len(vocab)
    print(f"Qwen2.5 vocabulary size: {vocab_size}")
    
    # Add elchat's special tokens to the tokenizer
    special_tokens_to_add = []
    for token in SPECIAL_TOKENS:
        if token not in vocab:
            special_tokens_to_add.append(token)
    
    if special_tokens_to_add:
        qwen_tokenizer.add_special_tokens({
            'additional_special_tokens': special_tokens_to_add
        })
        print(f"Added {len(special_tokens_to_add)} special tokens to tokenizer")
    
    # Save the tokenizer in HuggingFace format
    os.makedirs(output_dir, exist_ok=True)
    qwen_tokenizer.save_pretrained(output_dir)
    print(f"Saved HuggingFace tokenizer to {output_dir}")
    
    # Also save token_bytes tensor for loss evaluation
    import torch
    token_bytes_list = []
    for i in range(len(qwen_tokenizer)):
        try:
            token = qwen_tokenizer.decode([i])
            token_bytes_list.append(len(token.encode('utf-8')))
        except Exception:
            token_bytes_list.append(1)  # Default to 1 byte for special tokens
    
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.float32)
    token_bytes_path = os.path.join(output_dir, "token_bytes.pt")
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Saved token_bytes to {token_bytes_path}")
    
    # Save a marker file indicating this is a HuggingFace tokenizer
    marker_path = os.path.join(output_dir, "tokenizer_type.txt")
    with open(marker_path, "w") as f:
        f.write("huggingface\n")
        f.write(f"model: Qwen/Qwen2.5-0.5B\n")
        f.write(f"vocab_size: {len(qwen_tokenizer)}\n")
    
    # Test the tokenizer
    test_texts = [
        "Hola, ¿cómo estás?",
        "El rápido zorro marrón salta sobre el perro perezoso.",
        "¡Buenos días! ¿Qué tal?",
        "La inteligencia artificial está cambiando el mundo.",
    ]
    
    print("\nTokenizer test (Spanish):")
    for text in test_texts:
        ids = qwen_tokenizer.encode(text)
        decoded = qwen_tokenizer.decode(ids)
        print(f"  '{text}' -> {len(ids)} tokens -> '{decoded}'")
    
    print(f"\n✅ Tokenizer setup complete!")
    print(f"   Vocabulary size: {len(qwen_tokenizer)}")
    print(f"   Output directory: {output_dir}")
    
    return qwen_tokenizer

def train_spanish_tokenizer(output_dir: str, vocab_size: int = 65536, max_chars: int = 2_000_000_000):
    """
    Train a new BPE tokenizer on Spanish data.
    
    This uses rustbpe for training (same as original elchat).
    """
    print(f"Training Spanish tokenizer with vocab_size={vocab_size}...")
    
    import rustbpe
    
    base_dir = get_base_dir()
    data_dir = os.path.join(base_dir, "base_data")
    
    # Check if Spanish data exists
    import pyarrow.parquet as pq
    parquet_files = sorted([
        os.path.join(data_dir, f) for f in os.listdir(data_dir)
        if f.endswith('.parquet') and not f.endswith('.tmp')
    ])
    
    if not parquet_files:
        raise FileNotFoundError(
            f"No parquet files found in {data_dir}. "
            "Please run download_spanish_data.py first."
        )
    
    print(f"Found {len(parquet_files)} parquet files")
    
    # Create text iterator
    def text_iterator():
        chars_seen = 0
        for filepath in parquet_files:
            if chars_seen >= max_chars:
                break
            pf = pq.ParquetFile(filepath)
            for rg_idx in range(pf.num_row_groups):
                if chars_seen >= max_chars:
                    break
                rg = pf.read_row_group(rg_idx)
                texts = rg.column('text').to_pylist()
                for text in texts:
                    if chars_seen >= max_chars:
                        break
                    chars_seen += len(text)
                    yield text
        print(f"Processed {chars_seen:,} characters for tokenizer training")
    
    # Train using rustbpe
    tokenizer = rustbpe.Tokenizer()
    vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS)
    assert vocab_size_no_special >= 256
    
    print(f"Training tokenizer on up to {max_chars:,} characters...")
    tokenizer.train_from_iterator(text_iterator(), vocab_size_no_special, pattern=SPLIT_PATTERN)
    
    # Construct tiktoken encoding
    pattern = tokenizer.get_pattern()
    mergeable_ranks_list = tokenizer.get_mergeable_ranks()
    mergeable_ranks = {bytes(k): v for k, v in mergeable_ranks_list}
    tokens_offset = len(mergeable_ranks)
    special_tokens = {name: tokens_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
    
    enc = tiktoken.Encoding(
        name="elchat_spanish",
        pat_str=pattern,
        mergeable_ranks=mergeable_ranks,
        special_tokens=special_tokens,
    )
    
    # Save
    os.makedirs(output_dir, exist_ok=True)
    pickle_path = os.path.join(output_dir, "tokenizer.pkl")
    with open(pickle_path, "wb") as f:
        pickle.dump(enc, f)
    print(f"Saved tokenizer to {pickle_path}")
    
    # Save token_bytes tensor
    import torch
    token_bytes_list = []
    for i in range(enc.n_vocab):
        try:
            token_bytes = enc.decode_single_token_bytes(i)
            token_bytes_list.append(len(token_bytes))
        except Exception:
            token_bytes_list.append(1)
    
    token_bytes_tensor = torch.tensor(token_bytes_list, dtype=torch.float32)
    token_bytes_path = os.path.join(output_dir, "token_bytes.pt")
    torch.save(token_bytes_tensor, token_bytes_path)
    print(f"Saved token_bytes to {token_bytes_path}")
    
    # Test
    test_texts = [
        "Hola, ¿cómo estás?",
        "El rápido zorro marrón salta sobre el perro perezoso.",
    ]
    
    print("\nTokenizer test (Spanish):")
    for text in test_texts:
        ids = enc.encode(text)
        decoded = enc.decode(ids)
        chars_per_token = len(text) / len(ids)
        print(f"  '{text}' -> {len(ids)} tokens ({chars_per_token:.2f} chars/token)")
    
    print(f"\n✅ Tokenizer training complete!")
    print(f"   Vocabulary size: {enc.n_vocab}")
    
    return enc

def main():
    parser = argparse.ArgumentParser(description="Setup tokenizer for Spanish elchat")
    parser.add_argument(
        "--source",
        type=str,
        default="qwen2.5",
        choices=["qwen2.5", "train"],
        help="Tokenizer source: qwen2.5 (recommended) or train from Spanish data"
    )
    parser.add_argument(
        "--vocab_size",
        type=int,
        default=65536,
        help="Vocabulary size (only for --source train)"
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=2_000_000_000,
        help="Max characters for training (only for --source train)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ~/.cache/elchat/tokenizer)"
    )
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_dir = get_base_dir()
        output_dir = os.path.join(base_dir, "tokenizer")
    
    print("=" * 60)
    print("elchat Spanish Tokenizer Setup")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()
    
    if args.source == "qwen2.5":
        setup_qwen_tokenizer(output_dir)
    else:
        train_spanish_tokenizer(output_dir, args.vocab_size, args.max_chars)

if __name__ == "__main__":
    main()

