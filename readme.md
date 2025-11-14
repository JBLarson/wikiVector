# Wikipedia Embeddings Generation System


The code for pages_articles is a distinct system processing the largest dump, I used a GCP VM with a Tesla 4 to decrease job time from days (with my laptop) to hours, in November 2025 it cost me ~$14 and took several hours.


### Stage One - pages_articles



### Overview
This system generates semantic embeddings for all 6.9M English Wikipedia articles using a production-optimized pipeline with:
- Multi-core XML parsing
- GPU-accelerated batch encoding  
- Incremental FAISS index building
- Automatic checkpointing every 100K articles
- Comprehensive validation


## System Architecture
```mermaid
%% Renders best Left-to-Right
graph LR
    
    %% === PHASE 1: PRE-PROCESSING ===
    subgraph "PHASE 1: PRE-PROCESSING"
        direction TB
        A(fa:fa-file-archive enwiki.xml.bz2) -- "lbzip2 -d" --> B{parallel_decompression.sh}
        B -- "All Cores" --> C(fa:fa-file-alt enwiki.xml)
        C -- "Single-Core Read" --> D{chunk.py}
        D -- "Parallel Write Pool" --> E(fa:fa-folder-open XML Chunks)
    end

    %% === PHASE 2: EMBEDDING & CHECKPOINTING ===
    subgraph "PHASE 2: EMBEDDING PIPELINE"
        direction TB
        
        %% This subgraph represents the main script and its internal workers
        subgraph "generate_wiki_embeddings.py"
            direction LR
            P{"fa:fa-cogs Parser Workers (CPUs)"}
            Q["fa:fa-rocket GPU Encoder (T4)"]
            R{"fa:fa-database Index/DB Builder"}
            
            P -- "Article Batches" --> Q
            Q -- "Vector Batches" --> R
        end

        %% The outputs of the pipeline
        R -- "Saves every 100k articles" --> G((fa:fa-history Checkpoints))
        R -- "Saves final output" --> H((fa:fa-database metadata.db))
        R -- "Saves final output" --> I((fa:fa-sitemap index.faiss))
    end


    %% === Connections between phases ===
    E -- "Parallel Read (One per chunk)" --> P

    %% Styling (makes it look 'frontend')
    classDef data fill:#e6f7ff,stroke:#0050b3,stroke-width:2px
    classDef script fill:#f6ffed,stroke:#237804,stroke-width:2px
    classDef output fill:#fffbe6,stroke:#ad8b00,stroke-width:2px

    class A,C,E data
    class B,D script
    class P,Q,R script
    class G,H,I,K output
```





#### Download Wiki dump

aria2c -x 16 -s 16 "https://dumps.wikimedia.org/enwiki/20251101/enwiki-20251101-pages-articles-multistream.xml.bz2"



### Key Features

- **Resume Capability**: Automatic checkpoint detection and resume from last saved state
- **Memory Efficient**: Streaming XML parsing with element cleanup prevents OOM
- **GPU Optimized**: FP16 inference with batched encoding maximizes Tesla T4 utilization
- **Production Ready**: IVF+PQ compression reduces index size by ~8x with minimal accuracy loss
- **Fault Tolerant**: Checkpoint every 100K articles + graceful interrupt handling


## Stage Two - Enhance Vector DB

prepDb.py

getPageviews.py

processPageviews.py

processPagelinks.py


