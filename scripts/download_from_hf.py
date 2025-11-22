#!/usr/bin/env python3

"""
Download Wikipedia embeddings from Hugging Face Hub
"""

import os
import sys
import json
from pathlib import Path
from huggingface_hub import hf_hub_download
import argparse

# Configuration
HF_USERNAME = "jblarson3"  # Change this
REPO_NAME = "wiki-embeddings"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

FILES_TO_DOWNLOAD = [
    "index.faiss",
    "metadata.db",
]

def download_version(version: str = None):
    """Download artifacts from Hugging Face"""
    
    # Get version from manifest if not specified
    if not version:
        versions_file = Path("../data/versions.json")
        if versions_file.exists():
            with open(versions_file) as f:
                manifest = json.load(f)
                version = manifest.get("current")
                print(f"Using current version from manifest: {version}")
        else:
            version = "main"
            print(f"No version specified, using: {version}")
    
    print(f"\nDownloading Wikipedia embeddings {version}...")
    print(f"Repository: {REPO_ID}")
    print("-" * 60)
    
    # Create data directory
    os.makedirs("data", exist_ok=True)
    
    # Download each file
    for filename in FILES_TO_DOWNLOAD:
        print(f"\nDownloading {filename}...")
        
        try:
            downloaded_path = hf_hub_download(
                repo_id=REPO_ID,
                repo_type="dataset",
                filename=filename,
                revision=version,
                local_dir="data",
                local_dir_use_symlinks=False  # Actually copy the file
            )
            file_size_mb = os.path.getsize(downloaded_path) / (1024**2)
            print(f"  ✓ Downloaded to data/{filename} ({file_size_mb:.1f} MB)")
        except Exception as e:
            print(f"  ✗ Download failed: {e}")
            return False
    
    print("\n" + "=" * 60)
    print("✓ Download complete!")
    print("=" * 60)
    return True

def main():
    parser = argparse.ArgumentParser(description="Download Wikipedia embeddings from Hugging Face")
    parser.add_argument("--version", help="Version to download (default: current from manifest)")
    
    args = parser.parse_args()
    
    success = download_version(args.version)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()