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

# Setup Python environment
echo ""
echo "Setting up Python environment..."

# Install pip if needed
if ! command -v pip3 &> /dev/null; then
    echo "Installing pip..."
    sudo apt-get update
    sudo apt-get install -y python3-pip
fi

# Upgrade pip
pip3 install --upgrade pip

# Install dependencies
echo ""
echo "Installing Python dependencies..."
echo "This will take 2-3 minutes..."

pip3 install torch==2.1.0 --index-url https://download.pytorch.org/whl/cu118
pip3 install sentence-transformers==2.2.2
pip3 install faiss-gpu==1.7.2
pip3 install numpy==1.24.3
pip3 install tqdm==4.66.1
pip3 install psutil==5.9.5

echo "✓ Dependencies installed"

# Verify GPU is accessible from Python
echo ""
echo "Verifying PyTorch GPU access..."
python3 << EOF
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
else:
    print("WARNING: CUDA not available!")
EOF

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