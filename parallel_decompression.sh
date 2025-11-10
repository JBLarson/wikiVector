#!/bin/bash
# Parallel decompression of Wikipedia dump using pbzip2
# --- FIXED FOR MACOS/LINUX COMPATIBILITY ---

set -e

echo "=========================================="
echo "Wikipedia Dump Decompression (Parallel)"
echo "=========================================="
echo ""

#cd /mnt/data-large/wikipedia/raw
cd data


INPUT_FILE="enwiki-20251101-pages-articles-multistream.xml.bz2"
OUTPUT_FILE="enwiki-20251101-pages-articles-multistream.xml"

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: $INPUT_FILE not found"
    exit 1
fi

# Check if pbzip2 is installed (Cross-platform)
if ! command -v pbzip2 &> /dev/null; then
    echo "pbzip2 (parallel bzip2) not found."
    if [[ "$(uname)" == "Darwin" ]]; then
        echo "Attempting to install with Homebrew..."
        if ! command -v brew &> /dev/null; then
            echo "ERROR: Homebrew not found. Please install Homebrew first, or install pbzip2 manually."
            exit 1
        fi
        brew install pbzip2
    else
        echo "Attempting to install with apt-get (Linux)..."
        sudo apt-get update
        sudo apt-get install -y pbzip2
    fi
fi

# Check available space (Cross-platform)
echo "Checking file size and disk space..."
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS stat: -f "%z" = size in bytes
    COMPRESSED_SIZE=$(stat -f "%z" "$INPUT_FILE")
    # macOS df: -g = 1G-blocks
    AVAILABLE=$(df -g . | tail -1 | awk '{print $4}')
else
    # Linux stat: -c "%s" = size in bytes
    COMPRESSED_SIZE=$(stat -c "%s" "$INPUT_FILE")
    # Linux df: -BG = 1G-blocks
    AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')
fi

COMPRESSED_GB=$((COMPRESSED_SIZE / 1024 / 1024 / 1024))

echo "Compressed file: ${COMPRESSED_GB}GB"
echo "Available space: ${AVAILABLE}GB"
echo "Uncompressed will be ~113GB"
echo ""

if [ "$AVAILABLE" -lt 120 ]; then
    echo "WARNING: Less than 120GB free. May run out of space!"
    read -p "Continue anyway? (y/n) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
fi

# Get number of cores (Cross-platform)
if [[ "$(uname)" == "Darwin" ]]; then
    # macOS
    NUM_CORES=$(sysctl -n hw.ncpu)
else
    # Linux
    NUM_CORES=$(nproc)
fi

echo "Decompressing with pbzip2 using $NUM_CORES cores..."
echo "This will take ~5-10 minutes on a high-end server (much longer on a laptop)"
echo ""

time pbzip2 -d -k -p$NUM_CORES -v "$INPUT_FILE"

echo ""
echo "=========================================="
echo "Decompression Complete!"
echo "=========================================="
echo ""

# Verify output (Cross-platform)
if [ -f "$OUTPUT_FILE" ]; then
    if [[ "$(uname)" == "Darwin" ]]; then
        # macOS
        OUTPUT_SIZE=$(stat -f "%z" "$OUTPUT_FILE")
    else
        # Linux
        OUTPUT_SIZE=$(stat -c "%s" "$OUTPUT_FILE")
    fi
    
    OUTPUT_GB=$((OUTPUT_SIZE / 1024 / 1024 / 1024))
    echo "✓ Output file: $OUTPUT_FILE"
    echo "✓ Size: ${OUTPUT_GB}GB"
    ls -lh "$OUTPUT_FILE"
else
    echo "ERROR: Decompression failed"
    exit 1
fi