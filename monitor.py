#!/usr/bin/env python3
"""
Real-time monitoring dashboard for Wikipedia embeddings generation
Shows: progress, GPU utilization, memory usage, ETA, throughput
"""

import os
import sys
import time
import json
import subprocess
from pathlib import Path
from datetime import datetime, timedelta

try:
    import psutil
except ImportError:
    print("Installing psutil...")
    subprocess.run([sys.executable, "-m", "pip", "install", "psutil"], check=True)
    import psutil

def get_gpu_stats():
    """Get GPU utilization and memory from nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu',
             '--format=csv,noheader,nounits'],
            capture_output=True,
            text=True,
            check=True
        )
        
        parts = result.stdout.strip().split(', ')
        return {
            'gpu_util': int(parts[0]),
            'mem_used_mb': int(parts[1]),
            'mem_total_mb': int(parts[2]),
            'temperature': int(parts[3])
        }
    except Exception:
        return None

def get_latest_checkpoint(checkpoint_dir):
    """Find and parse the latest checkpoint"""
    checkpoint_path = Path(checkpoint_dir)
    
    if not checkpoint_path.exists():
        return None
    
    checkpoints = sorted(checkpoint_path.glob("checkpoint_*"))
    
    if not checkpoints:
        return None
    
    latest = checkpoints[-1]
    stats_file = latest / "stats.json"
    
    if not stats_file.exists():
        return None
    
    with open(stats_file) as f:
        stats = json.load(f)
    
    return stats

def format_time(seconds):
    """Format seconds to human readable"""
    if seconds < 60:
        return f"{seconds:.0f}s"
    elif seconds < 3600:
        return f"{seconds/60:.0f}m {seconds%60:.0f}s"
    else:
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        return f"{hours:.0f}h {minutes:.0f}m"

def clear_screen():
    """Clear terminal screen"""
    os.system('clear' if os.name == 'posix' else 'cls')

def print_dashboard(stats, gpu_stats, sys_stats):
    """Print monitoring dashboard"""
    clear_screen()
    
    print("=" * 80)
    print("WIKIPEDIA EMBEDDINGS GENERATION - LIVE MONITOR".center(80))
    print("=" * 80)
    print()
    
    # Progress
    if stats:
        articles_processed = stats.get('articles_processed', 0)
        total_articles = 6_900_000  # Approximate
        progress_pct = (articles_processed / total_articles) * 100
        
        elapsed = time.time() - stats.get('start_time', time.time())
        
        if articles_processed > 0:
            articles_per_sec = articles_processed / elapsed
            remaining = (total_articles - articles_processed) / articles_per_sec
            eta = datetime.now() + timedelta(seconds=remaining)
        else:
            articles_per_sec = 0
            eta = None
        
        print(f"PROGRESS")
        print(f"  Articles processed: {articles_processed:,} / {total_articles:,} ({progress_pct:.2f}%)")
        print(f"  Batches processed:  {stats.get('batches_processed', 0):,}")
        print(f"  Checkpoints saved:  {stats.get('checkpoints_saved', 0)}")
        print()
        
        # Performance
        print(f"PERFORMANCE")
        print(f"  Throughput:    {articles_per_sec:.0f} articles/sec ({articles_per_sec * 60:.0f}/min)")
        print(f"  Elapsed time:  {format_time(elapsed)}")
        if eta:
            print(f"  ETA:           {eta.strftime('%H:%M:%S')} ({format_time(remaining)} remaining)")
        print()
    else:
        print("PROGRESS")
        print("  No checkpoint data available yet...")
        print("  Pipeline may still be initializing.")
        print()
    
    # GPU stats
    if gpu_stats:
        print(f"GPU STATUS")
        print(f"  Utilization:   {gpu_stats['gpu_util']:3d}%")
        print(f"  Memory:        {gpu_stats['mem_used_mb']:,} MB / {gpu_stats['mem_total_mb']:,} MB " +
              f"({gpu_stats['mem_used_mb']/gpu_stats['mem_total_mb']*100:.1f}%)")
        print(f"  Temperature:   {gpu_stats['temperature']}°C")
        print()
    
    # System stats
    print(f"SYSTEM STATUS")
    print(f"  CPU usage:     {sys_stats['cpu_percent']:.1f}%")
    print(f"  RAM usage:     {sys_stats['mem_used_gb']:.1f} GB / {sys_stats['mem_total_gb']:.1f} GB " +
          f"({sys_stats['mem_percent']:.1f}%)")
    print(f"  Disk usage:    {sys_stats['disk_used_gb']:.1f} GB / {sys_stats['disk_total_gb']:.1f} GB " +
          f"({sys_stats['disk_percent']:.1f}%)")
    print()
    
    print("=" * 80)
    print(f"Last update: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("Press Ctrl+C to exit monitor (this will NOT stop the pipeline)")
    print("=" * 80)

def monitor(checkpoint_dir="/mnt/data/wikipedia/checkpoints", interval=5):
    """Main monitoring loop"""
    
    try:
        while True:
            # Get latest checkpoint stats
            stats = get_latest_checkpoint(checkpoint_dir)
            
            # Get GPU stats
            gpu_stats = get_gpu_stats()
            
            # Get system stats
            mem = psutil.virtual_memory()
            disk = psutil.disk_usage('/mnt/data')
            
            sys_stats = {
                'cpu_percent': psutil.cpu_percent(interval=1),
                'mem_used_gb': mem.used / (1024**3),
                'mem_total_gb': mem.total / (1024**3),
                'mem_percent': mem.percent,
                'disk_used_gb': disk.used / (1024**3),
                'disk_total_gb': disk.total / (1024**3),
                'disk_percent': disk.percent
            }
            
            # Print dashboard
            print_dashboard(stats, gpu_stats, sys_stats)
            
            # Wait
            time.sleep(interval)
            
    except KeyboardInterrupt:
        print("\n\nMonitor stopped.")
        sys.exit(0)

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Monitor Wikipedia embeddings generation")
    parser.add_argument(
        '--checkpoint-dir',
        default='/mnt/data/wikipedia/checkpoints',
        help='Path to checkpoints directory'
    )
    parser.add_argument(
        '--interval',
        type=int,
        default=5,
        help='Update interval in seconds'
    )
    
    args = parser.parse_args()
    
    print("Starting monitor...")
    print(f"Checkpoint directory: {args.checkpoint_dir}")
    print(f"Update interval: {args.interval}s")
    print()
    time.sleep(2)
    
    monitor(args.checkpoint_dir, args.interval)