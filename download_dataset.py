#!/usr/bin/env python3
"""
Helper script to download MMLU-Pro dataset from HuggingFace.
This will download the dataset to ./MMLU-Pro/data/
"""

import sys
from pathlib import Path

def download_with_datasets():
    """Download using the datasets library."""
    print("📥 Downloading MMLU-Pro dataset using datasets library...\n")
    
    try:
        from datasets import load_dataset
    except ImportError:
        print("❌ datasets library not installed.")
        print("Installing it now: pip3 install datasets")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "datasets"])
        from datasets import load_dataset
    
    # Load dataset
    print("Loading dataset from HuggingFace...")
    dataset = load_dataset('TIGER-Lab/MMLU-Pro')
    
    # Create output directory
    output_dir = Path(__file__).parent / 'MMLU-Pro' / 'data'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save splits
    print(f"\n💾 Saving to {output_dir}...")
    
    if 'validation' in dataset:
        val_path = output_dir / 'validation.parquet'
        dataset['validation'].to_parquet(val_path)
        print(f"✓ Saved validation set: {len(dataset['validation'])} questions")
    
    if 'test' in dataset:
        test_path = output_dir / 'test.parquet'
        dataset['test'].to_parquet(test_path)
        print(f"✓ Saved test set: {len(dataset['test'])} questions")
    
    print(f"\n✅ Dataset downloaded successfully!")
    print(f"📁 Location: {output_dir}")
    
    # List files
    print("\nDownloaded files:")
    for f in sorted(output_dir.glob('*.parquet')):
        size_mb = f.stat().st_size / (1024 * 1024)
        print(f"  - {f.name} ({size_mb:.1f} MB)")


def download_with_hub():
    """Download using huggingface_hub."""
    print("📥 Downloading MMLU-Pro dataset using huggingface_hub...\n")
    
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        print("❌ huggingface_hub not installed.")
        print("Installing it now: pip3 install huggingface-hub")
        import subprocess
        subprocess.run([sys.executable, "-m", "pip", "install", "huggingface-hub"])
        from huggingface_hub import snapshot_download
    
    output_dir = Path(__file__).parent / 'MMLU-Pro' / 'data'
    
    print(f"Downloading to {output_dir}...")
    snapshot_download(
        repo_id='TIGER-Lab/MMLU-Pro',
        repo_type='dataset',
        local_dir=str(output_dir),
        ignore_patterns=['*.md', '*.git*', 'README.md']
    )
    
    print(f"\n✅ Dataset downloaded successfully!")
    print(f"📁 Location: {output_dir}")


if __name__ == '__main__':
    print("=" * 60)
    print("MMLU-Pro Dataset Downloader")
    print("=" * 60)
    
    # Try datasets library first (recommended)
    try:
        download_with_datasets()
    except Exception as e:
        print(f"\n⚠️  datasets method failed: {e}")
        print("\nTrying alternative method with huggingface_hub...")
        try:
            download_with_hub()
        except Exception as e2:
            print(f"\n❌ Both download methods failed!")
            print(f"Error: {e2}")
            print("\nPlease manually download from:")
            print("https://huggingface.co/datasets/TIGER-Lab/MMLU-Pro")
            sys.exit(1)
