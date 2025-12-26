"""
Download and prepare Spanish pretraining data from FineWeb-2 or CulturaX.

This script downloads Spanish text data and converts it to the parquet format
expected by elchat's dataloader.

Usage:
    python -m scripts.download_spanish_data --source fineweb2 --num_shards 30
    python -m scripts.download_spanish_data --source culturax --num_shards 30

The data will be saved to ~/.cache/elchat/base_data/
"""

import os
import argparse
import time
from datasets import load_dataset
import pyarrow.parquet as pq
import pyarrow as pa

def get_base_dir():
    """Get the base directory for Spanish elchat data."""
    if os.environ.get("ELCHAT_BASE_DIR"):
        return os.environ.get("ELCHAT_BASE_DIR")
    home_dir = os.path.expanduser("~")
    cache_dir = os.path.join(home_dir, ".cache")
    # Use separate directory for Spanish data
    return os.path.join(cache_dir, "elchat")

def load_spanish_dataset(source: str, streaming: bool = True):
    """
    Load Spanish dataset from HuggingFace.
    
    Args:
        source: 'fineweb2' or 'culturax'
        streaming: Whether to use streaming mode (recommended for large datasets)
    
    Returns:
        HuggingFace dataset iterator
    """
    if source == "fineweb2":
        # FineWeb-2 Spanish subset
        # High quality, educationally filtered content
        print("Loading FineWeb-2 Spanish (spa_Latn)...")
        dataset = load_dataset(
            "HuggingFaceFW/fineweb-2",
            name="spa_Latn",  # Spanish Latin script
            split="train",
            streaming=streaming,
        )
    elif source == "culturax":
        # CulturaX Spanish
        # Good for Latin American and Peninsular Spanish
        print("Loading CulturaX Spanish...")
        dataset = load_dataset(
            "uonlp/CulturaX",
            "es",  # Spanish
            split="train",
            streaming=streaming,
        )
    elif source == "mc4":
        # mC4 Spanish (larger but lower quality)
        print("Loading mC4 Spanish...")
        dataset = load_dataset(
            "mc4",
            "es",
            split="train",
            streaming=streaming,
        )
    else:
        raise ValueError(f"Unknown source: {source}. Use 'fineweb2', 'culturax', or 'mc4'")
    
    return dataset

def convert_to_parquet_shards(
    dataset,
    output_dir: str,
    num_shards: int = 30,
    chars_per_shard: int = 250_000_000,
    row_group_size: int = 1024,
    text_column: str = "text",
):
    """
    Convert streaming dataset to parquet shards compatible with elchat.
    
    Args:
        dataset: HuggingFace dataset (streaming)
        output_dir: Directory to save parquet files
        num_shards: Maximum number of shards to create
        chars_per_shard: Target characters per shard (~250M = ~100MB compressed)
        row_group_size: Parquet row group size
        text_column: Name of the text column in dataset
    """
    os.makedirs(output_dir, exist_ok=True)
    
    shard_docs = []
    shard_index = 0
    shard_characters = 0
    total_docs_processed = 0
    t0 = time.time()
    
    print(f"Converting dataset to parquet shards...")
    print(f"Target: {num_shards} shards, ~{chars_per_shard:,} chars each")
    print(f"Output directory: {output_dir}")
    print()
    
    for doc in dataset:
        # Handle different dataset formats
        if text_column in doc:
            text = doc[text_column]
        elif "content" in doc:
            text = doc["content"]
        else:
            # Try to find any text field
            for key in ["text", "content", "document", "passage"]:
                if key in doc:
                    text = doc[key]
                    break
            else:
                print(f"Warning: Could not find text field in document. Keys: {doc.keys()}")
                continue
        
        if not isinstance(text, str) or len(text) < 100:
            continue  # Skip very short documents
            
        shard_docs.append(text)
        shard_characters += len(text)
        
        # Check if we should write a shard
        collected_enough_chars = shard_characters >= chars_per_shard
        docs_multiple_of_row_group_size = len(shard_docs) % row_group_size == 0
        
        if collected_enough_chars and docs_multiple_of_row_group_size:
            shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=row_group_size,
                use_dictionary=False,
                compression="zstd",
                compression_level=3,
                write_statistics=False,
            )
            
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            total_docs_processed += len(shard_docs)
            
            print(f"Wrote {shard_path}")
            print(f"  Documents: {len(shard_docs):,} | Characters: {shard_characters:,} | Time: {dt:.2f}s")
            
            shard_docs = []
            shard_characters = 0
            shard_index += 1
            
            # Stop if we've created enough shards
            if shard_index >= num_shards:
                print(f"\nReached target of {num_shards} shards. Stopping.")
                break
    
    # Write any remaining documents as the last shard
    if shard_docs and shard_index < num_shards:
        # Pad to row_group_size multiple if needed
        while len(shard_docs) % row_group_size != 0:
            shard_docs.append("")  # Add empty docs to pad
        
        shard_path = os.path.join(output_dir, f"shard_{shard_index:05d}.parquet")
        shard_table = pa.Table.from_pydict({"text": shard_docs})
        pq.write_table(
            shard_table,
            shard_path,
            row_group_size=row_group_size,
            use_dictionary=False,
            compression="zstd",
            compression_level=3,
            write_statistics=False,
        )
        print(f"Wrote final shard: {shard_path}")
        total_docs_processed += len(shard_docs)
        shard_index += 1
    
    print(f"\n✅ Done! Created {shard_index} shards with {total_docs_processed:,} documents")
    print(f"   Output directory: {output_dir}")
    return shard_index

def main():
    parser = argparse.ArgumentParser(description="Download Spanish pretraining data")
    parser.add_argument(
        "--source", 
        type=str, 
        default="fineweb2",
        choices=["fineweb2", "culturax", "mc4"],
        help="Data source: fineweb2 (recommended), culturax, or mc4"
    )
    parser.add_argument(
        "--num_shards", 
        type=int, 
        default=30,
        help="Number of shards to create (default: 30, ~7.5B chars)"
    )
    parser.add_argument(
        "--chars_per_shard",
        type=int,
        default=250_000_000,
        help="Characters per shard (default: 250M)"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: ~/.cache/elchat/base_data)"
    )
    args = parser.parse_args()
    
    # Set output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        base_dir = get_base_dir()
        output_dir = os.path.join(base_dir, "base_data")
    
    print("=" * 60)
    print("elchat Spanish Data Downloader")
    print("=" * 60)
    print(f"Source: {args.source}")
    print(f"Target shards: {args.num_shards}")
    print(f"Chars per shard: {args.chars_per_shard:,}")
    print(f"Output: {output_dir}")
    print("=" * 60)
    print()
    
    # Load dataset
    dataset = load_spanish_dataset(args.source, streaming=True)
    
    # Convert to parquet shards
    num_created = convert_to_parquet_shards(
        dataset,
        output_dir,
        num_shards=args.num_shards,
        chars_per_shard=args.chars_per_shard,
    )
    
    print(f"\n🎉 Successfully prepared {num_created} shards of Spanish data!")
    print(f"   Total estimated tokens: ~{num_created * args.chars_per_shard // 4:,}")

if __name__ == "__main__":
    main()

