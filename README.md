# TeletextSignals - Embeddings & RAG - Proof of Concept

This repository contains a **working proof of concept (PoC)** for building and querying semantic embeddings over short German news articles (Swiss Teletext), using modern transformer-based models and PostgreSQL with pgvector.

The system demonstrates:
- Document and chunk-level embedding ingestion
- Vector similarity search (cosine distance)
- Retrieval-Augmented Generation (RAG)-style retrieval
- Optional cross-encoder re-ranking for precision

---

## High-Level Intent

The PoC explores two complementary embedding strategies:

### 1. **Chunk-level embeddings for RAG**
- Model: `intfloat/multilingual-e5-large`
- Purpose:
    - High-recall semantic retrieval
    - Fine-grained chunk matching
    - RAG-style downstream usage
- Characteristics:
    - Text is chunked (≈800 characters with overlap)
    - Uses required E5 `query:` / `passage:` prefixes
    - 1024-dimensional vectors
    - Stored in PostgreSQL via pgvector

### 2. **Article-level embeddings for clustering / analytics**
- Model: `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
- Purpose:
    - Topic clustering
    - Article similarity
    - Exploratory analysis
- Characteristics:
    - One embedding per article (title + body)
    - 768-dimensional vectors
    - Stored in PostgreSQL via pgvector

---

## Retrieval & Reranking Flow

1. **Query embedding**
    - Model: `multilingual-e5-large`
    - Prefix: `query: <text>`
    - Normalized embeddings

2. **Vector retrieval**
    - PostgreSQL + pgvector
    - Cosine distance (`<=>`)
    - Top-K chunk retrieval

3. **Optional re-ranking**
    - Model: `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`
    - Input pairs:
      ```
      (query, title + chunk_text)
      ```
    - Produces higher-precision final ordering

This mirrors a standard **bi-encoder → cross-encoder** RAG architecture.

---

## Hardware Requirements

### Minimum (Proof of Concept)
- **GPU:** NVIDIA GPU with ≥ 4 GB VRAM
    - Tested successfully on **Quadro T2000 Max-Q**
- **CPU:** Any modern x86_64 CPU
- **RAM:** ≥ 16 GB recommended
- **Disk:** SSD strongly recommended (vector indexes)

### Notes
- `multilingual-e5-large` is GPU-heavy; batching is conservative by design.
- For lower VRAM environments, `multilingual-e5-small` might be an alternative (not tested)

---

## Software Stack

- Python 3.10+
- PostgreSQL 14+ with `pgvector`
- PyTorch
- Hugging Face Transformers
- Sentence-Transformers
- LangChain (HuggingFaceEmbeddings wrapper)
- psycopg2

---

## Key Design Decisions

- **Model alignments**
    - E5 uses `query:` / `passage:` prefixes
    - MPNet does not
- **Normalized embeddings**
    - Ensures stable cosine similarity
- **Separation of concerns**
    - RAG retrieval ≠ clustering embeddings
- **Database-first design**
    - PostgreSQL as the single source of truth
    - No external vector DB required

---