#!/bin/bash
#
# Setup script for Wikipedia embeddings generation
# Run this after SSH'ing into the GCP instance
#

set -e  # Exit on error

echo "=========================================="
echo "Wikipedia Embeddings - Environment Setup"
echo "=========================================="

# Check if running on correct instance
echo ""
echo "Checking GPU availability..."
if ! nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found. Is this a GPU instance?"
    exit 1
fi

nvidia-smi
echo ""

# Mount persistent disk (if not already mounted)
if ! mountpoint -q /mnt/data; then
    echo "Mounting persistent disk..."
    sudo mkfs.ext4 -F /dev/sdb 2>/dev/null || true
    sudo mkdir -p /mnt/data
    sudo mount /dev/sdb /mnt/data
    sudo chmod 777 /mnt/data
    echo "✓ Persistent disk mounted at /mnt/data"
else
    echo "✓ Persistent disk already mounted"
fi

# Create directory structure
echo ""
echo "Creating directory structure..."
mkdir -p /mnt/data/wikipedia/{raw,embeddings,checkpoints,logs}
echo "✓ Directories created"


echo ""
echo "=========================================="
echo "Setup complete!"
echo "=========================================="
echo ""
echo "Next steps:"
echo "1. Ensure Wikipedia dump download is complete"
echo "2. Run: python3 generate_wikipedia_embeddings.py"
echo ""
echo "The script will:"
echo "- Process ~6.9M articles in ~90 minutes"
echo "- Create checkpoints every 100K articles"
echo "- Output: /mnt/data/wikipedia/embeddings/"
echo ""