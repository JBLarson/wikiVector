import time
import requests
import sys
import os
from datetime import datetime, timedelta

# --- CONFIG ---
YEAR_MONTH = "2025-10" 
DOWNLOAD_DIR = "data/pageviews"
MAX_RETRIES = 5
RETRY_DELAY = 10
# ----------------

try:
    YEAR, MONTH = YEAR_MONTH.split('-')
except ValueError:
    print(f"Error: YEAR_MONTH format is invalid. Must be 'YYYY-MM'.")
    sys.exit(1)

BASE_URL = f"https://dumps.wikimedia.org/other/pageview_complete/{YEAR}/{YEAR_MONTH}"

SESSION = requests.Session()
SESSION.headers.update({
    'User-Agent': 'WikiExplorer-ETL/1.0 (Data download script)'
})

def get_day_urls(year, month):
    """Generates the list of daily 'user.bz2' URLs for the month."""
    urls = []
    month_int = int(month)
    year_int = int(year)
    
    if month_int == 12:
        last_day = 31
    else:
        last_day = (datetime(year_int, month_int + 1, 1) - timedelta(days=1)).day
        
    print(f"Generating URLs for {YEAR_MONTH} ({last_day} days)...")
    
    for day in range(1, last_day + 1):
        filename = f"pageviews-{year}{month}{day:02d}-user.bz2"
        urls.append((f"{BASE_URL}/{filename}", filename))
    return urls

def download_file(url, local_path):
    """
    Downloads a single file with retries and atomic writes.
    Supports resume by skipping already-downloaded files.
    """
    
    # Resume logic: skip if file already exists
    if os.path.exists(local_path):
        print(f"  [SKIP] File already exists: {os.path.basename(local_path)}")
        return True
    
    # Atomic write: download to .tmp file first
    tmp_path = local_path + ".tmp"
    
    # Clean up any leftover .tmp file from previous failed attempt
    if os.path.exists(tmp_path):
        print(f"  [CLEANUP] Removing incomplete temp file: {os.path.basename(tmp_path)}")
        os.remove(tmp_path)
        
    print(f"  [GET] Downloading {os.path.basename(local_path)}...")
    
    for attempt in range(MAX_RETRIES):
        try:
            with SESSION.get(url, stream=True, timeout=60) as res:
                if res.status_code >= 500:
                    wait_time = RETRY_DELAY * (attempt + 1)
                    print(f"  [WARN] Server error ({res.status_code}). Retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s...")
                    time.sleep(wait_time)
                    continue

                res.raise_for_status()
                
                # Download to .tmp file
                with open(tmp_path, 'wb') as f:
                    for chunk in res.iter_content(chunk_size=8192):
                        f.write(chunk)
                
                # Download complete - atomically rename to final path
                os.rename(tmp_path, local_path)
                print(f"  [OK] Saved {os.path.basename(local_path)}")
                return True
                
        except requests.RequestException as e:
            wait_time = RETRY_DELAY * (attempt + 1)
            print(f"  [WARN] Network error: {e}. Retry {attempt + 1}/{MAX_RETRIES} in {wait_time}s...")
            time.sleep(wait_time)
            
        except KeyboardInterrupt:
            print("\n  [CANCEL] Download cancelled by user. Cleaning up temp file...")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
            raise
            
    # Max retries exceeded - cleanup and fail
    print(f"  [FAIL] Max retries exceeded for {os.path.basename(local_path)}")
    if os.path.exists(tmp_path):
        os.remove(tmp_path)
    return False

def main():
    print(f"Starting pageview downloader for {YEAR_MONTH}")
    os.makedirs(DOWNLOAD_DIR, exist_ok=True)
    
    urls_to_fetch = get_day_urls(YEAR, MONTH)
    
    # Check how many files already exist
    existing_files = [f for url, f in urls_to_fetch if os.path.exists(os.path.join(DOWNLOAD_DIR, f))]
    if existing_files:
        print(f"Found {len(existing_files)}/{len(urls_to_fetch)} files already downloaded. Resuming...")
    
    try:
        succeeded = 0
        skipped = 0
        failed = 0
        
        for url, filename in urls_to_fetch:
            local_path = os.path.join(DOWNLOAD_DIR, filename)
            result = download_file(url, local_path)
            
            if os.path.exists(local_path) and result:
                if os.path.basename(local_path) in [os.path.basename(f) for f in existing_files]:
                    skipped += 1
                else:
                    succeeded += 1
            elif not result:
                failed += 1
        
        print(f"\n{'='*60}")
        print(f"Download complete!")
        print(f"  Newly downloaded: {succeeded}")
        print(f"  Already existed: {skipped}")
        print(f"  Failed: {failed}")
        print(f"{'='*60}")
        
        if failed > 0:
            print(f"\n[WARNING] {failed} file(s) failed to download. Run the script again to retry.")
            sys.exit(1)
            
    except KeyboardInterrupt:
        print("\n[STOP] Script interrupted. Run again to resume from where you left off.")
        sys.exit(0)

if __name__ == "__main__":
    main()