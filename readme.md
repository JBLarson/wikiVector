# Wikipedia Embeddings Generation System
## Production-Grade Pipeline

### Overview
This system generates semantic embeddings for all 6.9M English Wikipedia articles using a production-optimized pipeline with:
- Multi-core XML parsing
- GPU-accelerated batch encoding  
- Incremental FAISS index building
- Automatic checkpointing every 100K articles
- Comprehensive validation


## System Architecture
```mermaid
graph TD
    %% === Phase 1: Data Pre-processing ===
    subgraph "Phase 1: Data Pre-processing"
        A["enwiki.xml.bz2 (Dump)"]
        B("parallel_decompression.sh")
        C["enwiki.xml (~113GB)"]
        D("chunk.py")
        E["XML Chunks (folder)"]
        
        A -- "pbzip2 -d" --> B
        B --> C
        C -- "iterparse + parallel write" --> D
        D --> E
    end

    %% === Phase 2: Embedding Pipeline ===
    subgraph "Phase 2: Embedding Pipeline (generate_wiki_embeddings.py)"
        F("XML Parser Workers [Producers]")
        G{"Article Queue"}
        H("Main Consumer Process")
        I{{"GPU Model (all-MiniLM-L6-v2)"}}
        J("FAISSIndexBuilder")
        K["wikipedia_metadata.db"]
        L["wikipedia_embeddings.faiss"]
        
        F -- "Parsed Articles" --> G
        G -- "Batched Articles" --> H
        H -- "Text Batch" --> I
        I -- "Vector Batch" --> H
        H -- "Vectors + Metadata" --> J
        J -- "Write" --> K
        J -- "Write" --> L
    end

    %% === Phase 3: Search Application ===
    subgraph "Phase 3: Search Application (search.py)"
        M["User Query (text)"]
        N("search.py")
        O["Search Results"]
        
        M -- "Text" --> N
        N --> O
    end

    %% === Links Between Subgraphs ===
    E --> F
    N -- "1. Encode Query" --> I
    I -- "2. Query Vector" --> N
    N -- "3. Search Index" --> L
    L -- "4. Article IDs" --> N
    N -- "5. Lookup Metadata" --> K
    K -- "6. Article Titles" --> N
```
### Key Features

- **Resume Capability**: Automatic checkpoint detection and resume from last saved state
- **Memory Efficient**: Streaming XML parsing with element cleanup prevents OOM
- **GPU Optimized**: FP16 inference with batched encoding maximizes Tesla T4 utilization
- **Production Ready**: IVF+PQ compression reduces index size by ~8x with minimal accuracy loss
- **Fault Tolerant**: Checkpoint every 100K articles + graceful interrupt handling
