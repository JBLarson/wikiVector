# Wikipedia Embeddings Generation System
## Production-Grade Pipeline

### Overview
This system generates semantic embeddings for all 6.9M English Wikipedia articles using a production-optimized pipeline with:
- Multi-core XML parsing
- GPU-accelerated batch encoding  
- Incremental FAISS index building
- Automatic checkpointing every 100K articles
- Comprehensive validation

### Expected Performance
- **Processing time**: 75-90 minutes
- **Throughput**: 1800-2200 articles/second
- **Cost**: ~$0.75 (spot pricing)
- **Output size**: ~11GB compressed

---

## Quick Start

### 1. Setup Environment
```bash
# SSH into your GCP instance
gcloud compute ssh wikipedia-embeddings --zone=us-central1-a

# Run setup script
chmod +x setup.sh
./setup.sh
```

### 2. Start Processing
```bash
# In one terminal - run the pipeline
python3 generate_wikipedia_embeddings.py

# In another terminal - monitor progress
python3 monitor.py
```

### 3. Quick Validation (Optional)
Before processing all articles, you can test with just 10K:
```python
# Edit generate_wikipedia_embeddings.py
# Comment out: self.run_full_processing_phase()
# This runs only Phase 1 (quick validation)
```

---

## Architecture

### Pipeline Phases

**Phase 1: Quick Validation (2 minutes)**
- Process first 10,000 articles
- Validate embedding quality
- Test semantic search
- Verify GPU is working correctly

**Phase 2: Full Processing (75-90 minutes)**
- Stream parse entire Wikipedia XML dump
- Generate embeddings in batches of 512
- Build FAISS index incrementally
- Save checkpoints every 100K articles

**Phase 3: Index Optimization (20 minutes)**
- Convert to IVF+PQ index
- Reduces size by 4x
- 50x faster search
- 99%+ recall maintained

**Phase 4: Final Validation (5 minutes)**
- Run semantic search tests
- Benchmark query latency
- Generate validation report

### Key Components

**XMLStreamParser**
- Memory-efficient streaming parser
- Processes 23GB compressed file without decompression
- Filters non-article pages
- Cleans wiki markup

**EmbeddingGenerator**
- Uses `all-MiniLM-L6-v2` model (384 dimensions)
- FP16 mixed precision for 2x speedup
- Batch size 512 optimized for T4 GPU
- L2-normalized embeddings for cosine similarity

**FAISSIndexBuilder**
- Starts with flat index for development
- Converts to IVF+PQ for production
- Maintains SQLite metadata database
- Robust checkpointing

**ValidationSuite**
- Statistical validation (NaN/Inf detection)
- Semantic coherence tests
- Search quality benchmarks
- Coverage verification

---

## Output Files

```
/mnt/data/wikipedia/embeddings/
├── wikipedia_embeddings.faiss      # 10GB - Vector index
├── wikipedia_metadata.db           # 500MB - Article metadata
├── validation_report.json          # Quality metrics
└── logs/                          # Processing logs
    └── embeddings_YYYYMMDD_HHMMSS.log

/mnt/data/wikipedia/checkpoints/    # Deleted after success
├── checkpoint_00100000/
├── checkpoint_00200000/
└── ...
```

---

## Checkpointing & Recovery

### Automatic Checkpoints
- Saved every 100,000 articles (2-3 minutes)
- Contains:
  - Partial FAISS index
  - Metadata database snapshot
  - Processing statistics
  - Data integrity hash

### Recovery from Interruption
If the pipeline is interrupted (spot instance preempted, OOM, etc.):

```bash
# The pipeline will automatically detect the latest checkpoint
# and resume from there when you restart it
python3 generate_wikipedia_embeddings.py
```

You'll lose at most 100K articles of progress (≈5 minutes of work).

---

## Monitoring

### Real-time Dashboard
```bash
python3 monitor.py
```

Shows:
- Articles processed / total
- Throughput (articles/sec)
- ETA and elapsed time
- GPU utilization and memory
- System CPU/RAM/disk usage
- Checkpoints saved

### Manual Checks
```bash
# Check GPU utilization
nvidia-smi

# Check disk space
df -h /mnt/data

# Check latest checkpoint
ls -lh /mnt/data/wikipedia/checkpoints/

# View logs
tail -f /mnt/data/wikipedia/embeddings/logs/*.log
```

---

## Validation

### Built-in Tests

**Statistical Validation**
- Embedding magnitude ≈ 1.0 (normalized)
- No NaN or Inf values
- Expected mean/std distribution

**Semantic Validation**
- 5 hand-curated query/result pairs
- Expected recall@10 > 0.5
- Tests: programming languages, ML concepts, countries, physics, art

**Performance Validation**
- Search latency < 50ms (mean)
- Index size within expected bounds
- Coverage matches metadata DB

### Manual Testing

After completion, you can test the index:

```python
import faiss
import sqlite3
import numpy as np
from sentence_transformers import SentenceTransformer

# Load index
index = faiss.read_index("/mnt/data/wikipedia/embeddings/wikipedia_embeddings.faiss")

# Load metadata
db = sqlite3.connect("/mnt/data/wikipedia/embeddings/wikipedia_metadata.db")

# Load model
model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Search
query = "artificial intelligence and neural networks"
query_emb = model.encode([query], normalize_embeddings=True)
distances, indices = index.search(query_emb, k=10)

# Get titles
cursor = db.cursor()
for idx in indices[0]:
    cursor.execute("SELECT title FROM articles WHERE idx = ?", (int(idx),))
    print(cursor.fetchone()[0])
```

---

## Troubleshooting

### GPU Out of Memory
```python
# Reduce batch size in Config class
batch_size: int = 256  # Down from 512
```

### Slow Processing (< 1000 articles/sec)
- Check GPU utilization: should be 90-100%
- Verify FP16 is enabled
- Check disk I/O isn't bottlenecked

### "No module named X"
```bash
pip3 install -r requirements.txt
```

### Checkpoint Corruption
```bash
# Delete corrupted checkpoint
rm -rf /mnt/data/wikipedia/checkpoints/checkpoint_XXXXXXXX

# Pipeline will resume from previous valid checkpoint
```

---

## Cost Optimization

### Minimize Costs
1. **Use spot instances** (already configured)
2. **Delete after completion**:
   ```bash
   # Transfer outputs first
   gcloud compute scp wikipedia-embeddings:/mnt/data/wikipedia/embeddings/* . --zone=us-central1-a
   
   # Then delete everything
   gcloud compute instances delete wikipedia-embeddings --zone=us-central1-a
   gcloud compute disks delete wikipedia-data --zone=us-central1-a
   ```

3. **Clean up intermediate files**:
   ```bash
   # After successful completion
   rm -rf /mnt/data/wikipedia/checkpoints
   rm -rf /mnt/data/wikipedia/raw/*.bz2  # Keep if you want to re-run
   ```

### Estimated Costs
- Spot instance (2 hours): $0.70
- Persistent disk (1 day): $0.01
- **Total**: ~$0.75

---

## Distribution

### Package for Distribution
```bash
cd /mnt/data/wikipedia/embeddings

# Create tarball
tar -czf wikipedia_embeddings_v1.tar.gz \
    wikipedia_embeddings.faiss \
    wikipedia_metadata.db \
    validation_report.json

# Check size
ls -lh wikipedia_embeddings_v1.tar.gz
# Should be ~11GB
```

### Upload Options
1. **GitHub Release** (if <2GB per file, need to split)
2. **Hugging Face Hub** (recommended for ML datasets)
3. **Google Drive / Dropbox**
4. **Academic Torrents**
5. **IPFS** (decentralized)

---

## Advanced Usage

### Custom Embedding Model
```python
# In Config class
model_name: str = "sentence-transformers/bge-large-en-v1.5"
embedding_dim: int = 1024
```

### Different Index Type
```python
# In FAISSIndexBuilder._init_index()
# For exact search (slower, no compression):
self.index = faiss.IndexFlatIP(self.config.embedding_dim)

# For approximate search (current default after optimization):
# IndexIVFPQ - handled automatically in optimize_index()
```

### Process Subset
```python
# Limit articles processed
# In XMLStreamParser.stream_articles(), add:
if article_count > 100_000:
    break
```

---

## Performance Benchmarks

### Expected Metrics (T4 GPU)
- **Parsing**: 50MB/sec from compressed XML
- **Encoding**: 1800-2200 articles/sec
- **FAISS add**: 10,000 articles/sec
- **Search latency**: 20-50ms (optimized index)

### Bottlenecks
1. **GPU underutilized** → Increase batch size
2. **CPU at 100%** → Reduce num_workers
3. **Disk I/O slow** → Use faster SSD (gp3)

---

## FAQ

**Q: Can I pause and resume?**  
A: Yes! Stop with Ctrl+C, checkpoints are auto-saved. Restart the script to resume.

**Q: What if spot instance gets preempted?**  
A: Checkpoints survive. Create new instance, reattach disk, resume.

**Q: Can I run this locally?**  
A: Yes, but will take 10-20x longer without GPU. Edit device='cpu' in EmbeddingGenerator.

**Q: How much RAM needed?**  
A: 8GB minimum, 16GB recommended. The T4 instance has 16GB.

**Q: Can I use different Wikipedia dump?**  
A: Yes, update `xml_dump_path` in Config. Works with any MediaWiki XML export.

---

## Support

For issues:
1. Check logs: `/mnt/data/wikipedia/embeddings/logs/`
2. Check checkpoints: `/mnt/data/wikipedia/checkpoints/`
3. Verify GPU: `nvidia-smi`
4. Check disk space: `df -h /mnt/data`

---

## License

This pipeline is MIT licensed. Wikipedia content follows CC-BY-SA 3.0.