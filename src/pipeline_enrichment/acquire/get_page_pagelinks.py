import gzip
import shutil
import os
import time
import sys
import requests
from urllib.parse import urljoin

# --- CONFIG ---
DATA_DIR = "data"
BASE_URL = "https://dumps.wikimedia.org/enwiki/latest/"
FILES_TO_PROCESS = [
    "enwiki-latest-page.sql.gz",
    "enwiki-latest-pagelinks.sql.gz"
]
MAX_RETRIES = 3
RETRY_DELAY = 10
# ----------------

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'WikiExplorer-ETL/1.0 (Dump download script)'
})

def format_bytes(bytes_val):
    """Format bytes into human-readable format."""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if bytes_val < 1024.0:
            return f"{bytes_val:.2f} {unit}"
        bytes_val /= 1024.0
    return f"{bytes_val:.2f} PB"

def download_file(filename):
    """Download a file from Wikimedia dumps with progress."""
    url = urljoin(BASE_URL, filename)
    local_path = os.path.join(DATA_DIR, filename)
    
    if os.path.exists(local_path):
        print(f"[SKIP] File already exists: {local_path}")
        return True
    
    # Download to temp file first
    tmp_path = local_path + ".tmp"
    
    print(f"[DOWNLOAD] Fetching {filename} from {BASE_URL}")
    print(f"  URL: {url}")
    
    for attempt in range(MAX_RETRIES):
        try:
            with SESSION.get(url, stream=True, timeout=60) as response:
                if response.status_code >= 500:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"  [WARN] Server error ({response.status_code}). Retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s...")
                    time.sleep(wait_time)
                    continue
                
                response.raise_for_status()
                
                # Get file size if available
                total_size = int(response.headers.get('content-length', 0))
                
                if total_size:
                    print(f"  File size: {format_bytes(total_size)}")
                
                # Download with progress
                downloaded = 0
                start_time = time.time()
                last_update = start_time
                
                with open(tmp_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded += len(chunk)
                        
                        # Update progress every second
                        current_time = time.time()
                        if current_time - last_update >= 1.0:
                            elapsed = current_time - start_time
                            rate = downloaded / elapsed if elapsed > 0 else 0
                            
                            if total_size:
                                percent = (downloaded / total_size) * 100
                                print(f"  Progress: {percent:.1f}% ({format_bytes(downloaded)}/{format_bytes(total_size)}) @ {format_bytes(rate)}/s", end='\r')
                            else:
                                print(f"  Downloaded: {format_bytes(downloaded)} @ {format_bytes(rate)}/s", end='\r')
                            
                            last_update = current_time
                
                # Download complete - rename to final path
                os.rename(tmp_path, local_path)
                
                elapsed = time.time() - start_time
                final_size = os.path.getsize(local_path)
                avg_rate = final_size / elapsed if elapsed > 0 else 0
                
                print(f"\n  [OK] Downloaded {format_bytes(final_size)} in {elapsed:.1f}s (avg {format_bytes(avg_rate)}/s)")
                return True
                
        except requests.RequestException as e:
            wait_time = RETRY_DELAY * (attempt + 1)
            print(f"\n  [WARN] Network error: {e}")
            if attempt < MAX_RETRIES - 1:
                print(f"  Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                print(f"  [FAIL] Max retries exceeded")
                
        except KeyboardInterrupt:
            print("\n  [CANCEL] Download cancelled by user")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
    
    # Cleanup on failure
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    
    return False

def decompress_file(gz_path):
    """Stream-decompresses a .gz file, printing progress."""
    
    sql_path = gz_path[:-3]  # Remove the .gz
    
    if os.path.exists(sql_path):
        print(f"[SKIP] Decompressed file already exists: {sql_path}")
        return True
    
    if not os.path.exists(gz_path):
        print(f"[ERROR] Source file not found: {gz_path}")
        return False
    
    print(f"[DECOMPRESS] {gz_path} -> {sql_path}")
    print("  This will take a while and use significant disk space...")
    
    start_time = time.time()
    
    try:
        gz_size = os.path.getsize(gz_path)
        processed = 0
        last_update = start_time
        
        with gzip.open(gz_path, 'rb') as f_in:
            with open(sql_path, 'wb') as f_out:
                while True:
                    chunk = f_in.read(16*1024*1024)  # 16MB buffer
                    if not chunk:
                        break
                    
                    f_out.write(chunk)
                    processed += len(chunk)
                    
                    # Update progress every 5 seconds
                    current_time = time.time()
                    if current_time - last_update >= 5.0:
                        elapsed = current_time - start_time
                        rate = processed / elapsed if elapsed > 0 else 0
                        percent = (f_in.tell() / gz_size) * 100 if gz_size > 0 else 0
                        
                        print(f"  Progress: {percent:.1f}% - Decompressed: {format_bytes(processed)} @ {format_bytes(rate)}/s", end='\r')
                        last_update = current_time
        
        end_time = time.time()
        elapsed = end_time - start_time
        
        gz_size_gb = gz_size / (1024**3)
        sql_size = os.path.getsize(sql_path)
        sql_size_gb = sql_size / (1024**3)
        
        print(f"\n  [OK] Decompressed in {elapsed:.1f}s")
        print(f"  {format_bytes(gz_size)} -> {format_bytes(sql_size)}")
        return True
        
    except Exception as e:
        print(f"\n[ERROR] Failed to decompress {gz_path}: {e}", file=sys.stderr)
        # Clean up partial file on error
        if os.path.exists(sql_path):
            os.remove(sql_path)
        return False

def main():
    print("Wikipedia Dump Downloader & Decompressor")
    print("=" * 60)
    
    # Ensure data directory exists
    os.makedirs(DATA_DIR, exist_ok=True)
    
    try:
        for filename in FILES_TO_PROCESS:
            print(f"\nProcessing: {filename}")
            print("-" * 60)
            
            # Step 1: Download if needed
            gz_path = os.path.join(DATA_DIR, filename)
            if not os.path.exists(gz_path):
                if not download_file(filename):
                    print(f"[ERROR] Failed to download {filename}")
                    continue
            else:
                print(f"[SKIP] Compressed file already exists: {gz_path}")
            
            # Step 2: Decompress if needed
            decompress_file(gz_path)
            
        print("\n" + "=" * 60)
        print("All files processed successfully!")
        
        # Show summary
        print("\nFiles in data directory:")
        for filename in FILES_TO_PROCESS:
            gz_path = os.path.join(DATA_DIR, filename)
            sql_path = gz_path[:-3]
            
            if os.path.exists(gz_path):
                gz_size = os.path.getsize(gz_path)
                print(f"  {filename}: {format_bytes(gz_size)}")
            
            if os.path.exists(sql_path):
                sql_size = os.path.getsize(sql_path)
                print(f"  {os.path.basename(sql_path)}: {format_bytes(sql_size)}")
        
    except KeyboardInterrupt:
        print("\n\n[STOP] Process interrupted by user")
        print("Run the script again to resume - already downloaded/decompressed files will be skipped")
        sys.exit(0)

if __name__ == "__main__":
    main()