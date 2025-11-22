#!/bin/bash

# --- Configuration ---
# NOTE: The full Wikipedia dump is provided as a compressed .bz2 file.
# The URL below targets this compressed file.
DUMP_DATE="20251101"
# We will download the compressed BZ2 file.
DUMP_FILE_COMPRESSED="enwiki-${DUMP_DATE}-pages-articles-multistream.xml.bz2"
DUMP_URL="https://dumps.wikimedia.org/enwiki/${DUMP_DATE}/${DUMP_FILE_COMPRESSED}"
TARGET_DIR="../data/raw"
TARGET_PATH="${TARGET_DIR}/${DUMP_FILE_COMPRESSED}"

# --- Execution ---

echo "=========================================="
echo "WIKIPEDIA DUMP DOWNLOADER (using aria2c)"
echo "=========================================="

# Check for aria2c availability
if ! command -v aria2c &> /dev/null
then
    echo "❌ ERROR: 'aria2c' is not installed or not in your PATH."
    echo "Please install it (e.g., 'brew install aria2' on macOS or 'sudo apt install aria2' on Debian/Ubuntu)."
    exit 1
fi

# 1. Create target directory
if [ ! -d "$TARGET_DIR" ]; then
    echo "Creating directory: ${TARGET_DIR}"
    mkdir -p "$TARGET_DIR"
else
    echo "Target directory already exists: ${TARGET_DIR}"
fi

# 2. Check if the compressed file already exists
if [ -f "$TARGET_PATH" ]; then
    echo ""
    echo "✅ Compressed dump already exists: ${TARGET_PATH}"
    echo "Skipping download."
    exit 0
fi

# 3. Download the file using aria2c
echo ""
echo "Starting parallel download for ${DUMP_FILE_COMPRESSED}..."
echo "Source: ${DUMP_URL}"
echo "Destination: ${TARGET_PATH}"
echo "------------------------------------------"

# -x 16: Max connections (16 streams)
# -k 1M: Min chunk size (1MB chunks)
# -c: Continue interrupted download
# -d: Directory to save to
# -o: Filename
aria2c -x 16 -k 1M -c -d "$TARGET_DIR" -o "$DUMP_FILE_COMPRESSED" "$DUMP_URL"

# 4. Check for success
if [ $? -eq 0 ]; then
    echo ""
    echo "=========================================="
    echo "✅ Download complete!"
    echo "Saved to: ${TARGET_PATH}"
    echo ""
    echo "NOTE: This is the compressed (.bz2) file. You'll need to decompress it"
    echo "to get the final .xml dump before chunking/parsing begins."
    echo "=========================================="
else
    echo ""
    echo "=========================================="
    echo "❌ ERROR: aria2c download failed."
    echo "=========================================="
    exit 1
fi
