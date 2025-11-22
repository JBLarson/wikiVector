# Wikipedia Semantic Search Engine

This project contains a complete, production-grade pipeline to build a semantic search engine from a raw English Wikipedia dump. The system is broken into two main stages, followed by a powerful, database-driven testing suite.

1.  **Stage 1: Embeddings Pipeline:** Ingests the 100GB+ XML dump, extracts 6.7M+ articles, generates vector embeddings (using `all-MiniLM-L6-v2`), and builds a compressed FAISS index and a `metadata.db`.
2.  **Stage 2: Metadata Enrichment:** Ingests the 30GB+ `pagelinks.sql` dump and daily pageview data to enrich the `metadata.db` with popularity and importance signals (backlinks, pageviews) for advanced re-ranking.
3.  **API & Testing Suite:** A Flask server to query the final database and a comprehensive testing suite to track performance and quality regressions over time.

---

## 🏛️ System Architecture

This diagram shows the high-level flow of data through the three main components of the project.
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


