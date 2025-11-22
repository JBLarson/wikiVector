#!/usr/bin/env python3
"""
Pre-flight validation script
Tests environment and dependencies before running full pipeline
Run this first to catch issues early
"""

import sys
import subprocess
from pathlib import Path

def print_header(text):
    print("\n" + "=" * 60)
    print(f"  {text}")
    print("=" * 60)

def check_python_version():
    """Verify Python version"""
    print("\n✓ Checking Python version...")
    version = sys.version_info
    print(f"  Python {version.major}.{version.minor}.{version.micro}")
    
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        print("  ❌ ERROR: Python 3.8+ required")
        return False
    
    return True

def check_gpu():
    """Verify GPU is accessible"""
    print("\n✓ Checking GPU...")
    
    try:
        result = subprocess.run(
            ['nvidia-smi'],
            capture_output=True,
            text=True,
            check=True
        )
        
        # Parse output for GPU name
        for line in result.stdout.split('\n'):
            if 'Tesla' in line or 'T4' in line or 'GPU' in line:
                print(f"  {line.strip()}")
                break
        
        return True
        
    except (subprocess.CalledProcessError, FileNotFoundError):
        print("  ❌ ERROR: nvidia-smi failed. Is GPU available?")
        return False

def check_pytorch():
    """Verify PyTorch and CUDA"""
    print("\n✓ Checking PyTorch...")
    
    try:
        import torch
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  GPU: {torch.cuda.get_device_name(0)}")
            
            # Test tensor on GPU
            x = torch.randn(10, 10).cuda()
            y = x @ x.t()
            
            print("  ✓ GPU tensor operations working")
            return True
        else:
            print("  ❌ ERROR: CUDA not available in PyTorch")
            return False
            
    except ImportError:
        print("  ❌ ERROR: PyTorch not installed")
        print("  Run: pip3 install torch --index-url https://download.pytorch.org/whl/cu118")
        return False
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

def check_dependencies():
    """Verify all required packages"""
    print("\n✓ Checking dependencies...")
    
    required = [
        ('sentence_transformers', 'sentence-transformers'),
        ('faiss', 'faiss-gpu'),
        ('numpy', 'numpy'),
        ('tqdm', 'tqdm'),
    ]
    
    all_good = True
    
    for module, package in required:
        try:
            __import__(module)
            print(f"  ✓ {package}")
        except ImportError:
            print(f"  ❌ {package} not installed")
            print(f"     Run: pip3 install {package}")
            all_good = False
    
    return all_good

def check_faiss_gpu():
    """Verify FAISS GPU support"""
    print("\n✓ Checking FAISS GPU support...")
    
    try:
        import faiss
        
        # Try to create GPU resources
        res = faiss.StandardGpuResources()
        
        # Create a simple index on GPU
        index_cpu = faiss.IndexFlatL2(128)
        index_gpu = faiss.index_cpu_to_gpu(res, 0, index_cpu)
        
        print("  ✓ FAISS GPU support working")
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: FAISS GPU failed: {e}")
        print("  Make sure faiss-gpu (not faiss-cpu) is installed")
        return False

def check_disk_space():
    """Verify sufficient disk space"""
    print("\n✓ Checking disk space...")
    
    data_dir = Path("/mnt/data-large")
    
    if not data_dir.exists():
        print("  ❌ ERROR: /mnt/data-large not mounted")
        print("  Run setup.sh to mount persistent disk")
        return False
    
    # Get disk usage
    stat = data_dir.stat()
    
    try:
        import psutil
        disk = psutil.disk_usage(str(data_dir))
        
        free_gb = disk.free / (1024**3)
        total_gb = disk.total / (1024**3)
        
        print(f"  Available: {free_gb:.1f} GB / {total_gb:.1f} GB")
        
        if free_gb < 50:
            print("  ⚠️  WARNING: Less than 50GB free")
            print("  Recommended: 100GB+ for full pipeline")
            return True  # Warning but not fatal
        
        return True
        
    except ImportError:
        print("  ⚠️  Cannot check disk space (psutil not installed)")
        return True

def check_wikipedia_dump():
    """Check if Wikipedia dump exists"""
    print("\n✓ Checking Wikipedia dump...")
    
    dump_path = Path("/mnt/data-large/wikipedia/raw/enwiki-20251101-pages-articles-multistream.xml")
    
    if not dump_path.exists():
        print(f"  ❌ ERROR: Wikipedia dump not found")
        print(f"  Expected: {dump_path}")
        print("  Make sure aria2c download completed successfully")
        return False
    
    size_gb = dump_path.stat().st_size / (1024**3)
    print(f"  ✓ Found dump: {size_gb:.1f} GB")
    
    if size_gb < 20:
        print("  ⚠️  WARNING: Dump seems too small")
        print("  Expected: ~23GB compressed")
        return True  # Warning but not fatal
    
    return True


def test_embedding_generation():
    """Quick test of embedding generation"""
    print("\n✓ Testing embedding generation...")
    
    try:
        from sentence_transformers import SentenceTransformer
        import time
        
        model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
        
        # Test single encoding
        start = time.time()
        emb = model.encode(["This is a test"], normalize_embeddings=True)
        single_time = time.time() - start
        
        print(f"  ✓ Single encoding: {single_time*1000:.1f}ms")
        
        # Test batch encoding
        texts = ["Test sentence"] * 100
        start = time.time()
        embs = model.encode(texts, batch_size=50, normalize_embeddings=True)
        batch_time = time.time() - start
        
        throughput = len(texts) / batch_time
        
        print(f"  ✓ Batch encoding: {batch_time*1000:.1f}ms for 100 texts")
        print(f"  ✓ Throughput: {throughput:.0f} texts/sec")
        
        if throughput < 100:
            print("  ⚠️  WARNING: Low throughput (check GPU utilization)")
        
        return True
        
    except Exception as e:
        print(f"  ❌ ERROR: {e}")
        return False

def main():
    """Run all pre-flight checks"""
    
    print_header("PRE-FLIGHT VALIDATION")
    print("This will verify your environment is ready for Wikipedia embeddings generation")
    
    checks = [
        ("Python version", check_python_version),
        ("GPU availability", check_gpu),
        ("PyTorch + CUDA", check_pytorch),
        ("Dependencies", check_dependencies),
        ("FAISS GPU", check_faiss_gpu),
        ("Disk space", check_disk_space),
        ("Wikipedia dump", check_wikipedia_dump),
        ("Embedding generation", test_embedding_generation),
    ]
    
    results = []
    
    for name, check_func in checks:
        try:
            result = check_func()
            results.append((name, result))
        except Exception as e:
            print(f"\n❌ ERROR in {name}: {e}")
            results.append((name, False))
    
    # Summary
    print_header("SUMMARY")
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for name, result in results:
        status = "✓ PASS" if result else "❌ FAIL"
        print(f"  {status}: {name}")
    
    print(f"\n  {passed}/{total} checks passed")
    
    if passed == total:
        print("\n  🎉 All checks passed!")
        print("  Ready to run: python3 generate_wikipedia_embeddings.py")
        return 0
    else:
        print("\n  ⚠️  Some checks failed. Please fix issues above.")
        return 1

if __name__ == "__main__":
    sys.exit(main())