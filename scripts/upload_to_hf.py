#!/usr/bin/env python3
"""
Upload Wikipedia embeddings artifacts to Hugging Face Hub

import os
import sys
import json
from pathlib import Path
from datetime import datetime
"""
from huggingface_hub import HfApi, create_repo
import argparse

# Configuration
HF_USERNAME = "jblarson3"  # Change this to your HF username
REPO_NAME = "wiki-embeddings"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

# Files to upload
FILES_TO_UPLOAD = [
    "../data/index.faiss",
    "../data/metadata.db",
]

def upload_version(version: str, notes: str = ""):
    """Upload artifacts to Hugging Face with versioning"""
    
    api = HfApi()
    
    # Ensure repo exists
    try:
        create_repo(
            repo_id=REPO_ID,
            repo_type="dataset",
            exist_ok=True
        )
        print(f"✓ Repository {REPO_ID} ready")
    except Exception as e:
        print(f"✗ Failed to create repo: {e}")
        return False
    
    print(f"\nUploading version {version}...")
    print("-" * 60)
    
    # Upload each file
    for filepath in FILES_TO_UPLOAD:
        if not os.path.exists(filepath):
            print(f"✗ File not found: {filepath}")
            continue
        
        file_size_mb = os.path.getsize(filepath) / (1024**2)
        print(f"\nUploading {filepath} ({file_size_mb:.1f} MB)...")
        
        try:
            api.upload_file(
                path_or_fileobj=filepath,
                path_in_repo=os.path.basename(filepath),
                repo_id=REPO_ID,
                repo_type="dataset",
                commit_message=f"Upload {os.path.basename(filepath)} - {version}",
            )
            print(f"  ✓ Uploaded successfully")
        except Exception as e:
            print(f"  ✗ Upload failed: {e}")
            return False
    
    # Create version tag
    try:
        api.create_tag(
            repo_id=REPO_ID,
            repo_type="dataset",
            tag=version,
            tag_message=notes or f"Version {version}",
        )
        print(f"\n✓ Created tag: {version}")
    except Exception as e:
        print(f"✗ Failed to create tag: {e}")
        # Non-fatal, continue
    
    # Update versions.json locally
    update_versions_manifest(version, notes)
    
    print("\n" + "=" * 60)
    print(f"✓ Upload complete!")
    print(f"  Repository: https://huggingface.co/datasets/{REPO_ID}")
    print(f"  Version: {version}")
    print("=" * 60)
    
    return True

def update_versions_manifest(version: str, notes: str):
    """Update local versions.json manifest"""
    
    versions_file = Path("data/versions.json")
    
    if versions_file.exists():
        with open(versions_file) as f:
            manifest = json.load(f)
    else:
        manifest = {"current": version, "versions": {}}
    
    # Add new version
    manifest["versions"][version] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hf_repo": REPO_ID,
        "tag": version,
        "notes": notes
    }
    manifest["current"] = version
    
    # Save
    with open(versions_file, 'w') as f:
        json.dump(manifest, f, indent=2)
    
    print(f"\n✓ Updated {versions_file}")

def main():
    parser = argparse.ArgumentParser(description="Upload Wikipedia embeddings to Hugging Face")
    parser.add_argument("--version", required=True, help="Version tag (e.g., v1.0.0)")
    parser.add_argument("--notes", default="", help="Release notes")
    
    args = parser.parse_args()
    
    # Validate version format
    if not args.version.startswith('v'):
        print("Version should start with 'v' (e.g., v1.0.0)")
        sys.exit(1)
    
    # Check files exist
    missing = [f for f in FILES_TO_UPLOAD if not os.path.exists(f)]
    if missing:
        print("Missing files:")
        for f in missing:
            print(f"  ✗ {f}")
        sys.exit(1)
    
    # Upload
    success = upload_version(args.version, args.notes)
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()