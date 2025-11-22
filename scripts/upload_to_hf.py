#!/usr/bin/env python3
"""
Upload embeddings artifacts to Hugging Face Hub with auto-versioning.
"""

import os
import json
from datetime import datetime
from pathlib import Path
from huggingface_hub import HfApi, create_repo

# ---------------------------------------------------------------------
# CONFIG
# ---------------------------------------------------------------------

HF_USERNAME = "jblarson3"
REPO_NAME = "wiki-embeddings"
REPO_ID = f"{HF_USERNAME}/{REPO_NAME}"

DATA_DIR = Path("data")

FILES_TO_UPLOAD = [
    DATA_DIR / "metadata.db",
    DATA_DIR / "versions.json",
]

VERSIONS_FILE = DATA_DIR / "versions.json"

# ---------------------------------------------------------------------
# VERSIONING LOGIC
# ---------------------------------------------------------------------

def load_manifest():
    """Load or initialize versions.json."""
    if VERSIONS_FILE.exists():
        with open(VERSIONS_FILE) as f:
            return json.load(f)
    
    # Initialize if does not exist
    return {
        "current": None,
        "versions": {}
    }

def determine_next_version(manifest):
    """Auto-increment semantic versions: v0.0.2 → v0.0.3"""
    
    current = manifest.get("current")
    
    # If no version exists yet
    if not current:
        return "v0.0.1"

    if not current.startswith("v"):
        raise ValueError(f"Invalid version format: {current}")

    parts = current[1:].split(".")
    if len(parts) != 3 or not all(p.isdigit() for p in parts):
        raise ValueError(f"Invalid semantic version format: {current}")

    major, minor, patch = map(int, parts)

    # Increment PATCH by default
    patch += 1

    return f"v{major}.{minor}.{patch}"



# ---------------------------------------------------------------------
# UPLOAD LOGIC
# ---------------------------------------------------------------------

def upload_version(version, notes=""):
    api = HfApi()

    print(f"Preparing to upload version {version}...")
    
    # Ensure repo exists
    create_repo(repo_id=REPO_ID, repo_type="dataset", exist_ok=True)

    # Upload files
    for f in FILES_TO_UPLOAD:
        if not f.exists():
            print(f"✗ Missing file: {f}")
            continue

        print(f"Uploading: {f}...")
        api.upload_file(
            path_or_fileobj=str(f),
            path_in_repo=f.name,
            repo_id=REPO_ID,
            repo_type="dataset",
            commit_message=f"Upload {f.name} ({version})"
        )

    # Tag version
    print("Creating tag on HF...")
    api.create_tag(
        repo_id=REPO_ID,
        repo_type="dataset",
        tag=version,
        tag_message=notes or f"Version {version}"
    )

    print(f"✓ Uploaded and tagged version {version}")

# ---------------------------------------------------------------------
# MANIFEST UPDATE
# ---------------------------------------------------------------------

def update_manifest(manifest, version, notes):
    manifest["versions"][version] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "hf_repo": REPO_ID,
        "tag": version,
        "notes": notes
    }
    manifest["current"] = version

    with open(VERSIONS_FILE, "w") as f:
        json.dump(manifest, f, indent=2)
    
    print(f"✓ Updated manifest at {VERSIONS_FILE}")

# ---------------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------------

def main():
    manifest = load_manifest()
    next_version = determine_next_version(manifest)
    notes = f"Automatic upload for {next_version}"

    upload_version(next_version, notes)
    update_manifest(manifest, next_version, notes)

    print("\nDone!")
    print(f"Version created: {next_version}")
    print(f"Repo: https://huggingface.co/datasets/{REPO_ID}")

if __name__ == "__main__":
    main()
