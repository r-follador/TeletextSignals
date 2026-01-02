\connect teletext

-- Enable pgvector for similarity search.
CREATE EXTENSION IF NOT EXISTS vector;

-------------------------------------------
CREATE TABLE IF NOT EXISTS docs_teletext (
    id BIGSERIAL PRIMARY KEY,
    teletext_id TEXT NOT NULL,
    title TEXT,
    content TEXT,
    publication_datetime TIMESTAMPTZ,
    rubric TEXT,
    categories TEXT[],
    language TEXT,
    fetched_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_teletext_teletext_id
    ON docs_teletext (teletext_id);

CREATE INDEX IF NOT EXISTS idx_docs_teletext_pub_datetime
    ON docs_teletext (publication_datetime);

CREATE TABLE IF NOT EXISTS emb_teletext_chunk (
    id BIGSERIAL PRIMARY KEY,
    teletext_id TEXT NOT NULL REFERENCES docs_teletext (teletext_id),
    chunk_id INTEGER,
    chunk_text TEXT,
    embedding_em5 vector(1024)
);

CREATE INDEX IF NOT EXISTS idx_rag_teletext_emb1024_cosine
    ON emb_teletext_chunk
        USING ivfflat (embedding_em5 vector_cosine_ops)
    WITH (lists = 100);

CREATE INDEX IF NOT EXISTS idx_rag_teletext_teletext_id
    ON emb_teletext_chunk (teletext_id);

CREATE UNIQUE INDEX IF NOT EXISTS idx_rag_teletext_source_chunk
    ON emb_teletext_chunk (teletext_id, chunk_id);
