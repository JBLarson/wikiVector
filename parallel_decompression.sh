#!/bin/bash
# Parallel decompression of Wikipedia dump using pbzip2

set -e

echo "=========================================="
echo "Wikipedia Dump Decompression (Parallel)"
echo "=========================================="
echo ""

cd /mnt/data-large/wikipedia/raw

INPUT_FILE="enwiki-20251101-pages-articles-multistream.xml.bz2"
OUTPUT_FILE="enwiki-20251101-pages-articles-multistream.xml"

# Check if input exists
if [ ! -f "$INPUT_FILE" ]; then
    echo "ERROR: $INPUT_FILE not found"
    exit 1
fi

# Check if pbzip2 is installed
if ! command -v pbzip2 &> /dev/null; then
    echo "Installing pbzip2 (parallel bzip2)..."
    sudo apt-get update
    sudo apt-get install -y pbzip2
fi

# Check available space
COMPRESSED_SIZE=$(stat -f --format="%s" "$INPUT_FILE" 2>/dev/null || stat -c "%s" "$INPUT_FILE")
COMPRESSED_GB=$((COMPRESSED_SIZE / 1024 / 1024 / 1024))
AVAILABLE=$(df -BG . | tail -1 | awk '{print $4}' | sed 's/G//')

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

# Decompress with all CPU cores
NUM_CORES=$(nproc)
echo "Decompressing with pbzip2 using $NUM_CORES cores..."
echo "This will take ~5-10 minutes"
echo ""

time pbzip2 -d -k -p$NUM_CORES -v "$INPUT_FILE"

echo ""
echo "=========================================="
echo "Decompression Complete!"
echo "=========================================="
echo ""

# Verify output
if [ -f "$OUTPUT_FILE" ]; then
    OUTPUT_SIZE=$(stat -c "%s" "$OUTPUT_FILE")
    OUTPUT_GB=$((OUTPUT_SIZE / 1024 / 1024 / 1024))
    echo "✓ Output file: $OUTPUT_FILE"
    echo "✓ Size: ${OUTPUT_GB}GB"
    ls -lh "$OUTPUT_FILE"
else
    echo "ERROR: Decompression failed"
    exit 1
fi

echo ""
echo "Next step: Run chunking script"
echo ""