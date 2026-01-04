\connect teletext

-- Enable pgvector for similarity search.
CREATE EXTENSION IF NOT EXISTS vector;

SET maintenance_work_mem = '512MB';   -- more memory required for ivfflat index creation

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
    fetched_at TIMESTAMPTZ DEFAULT NOW(),
    fts tsvector
        GENERATED ALWAYS AS (
            to_tsvector('german', coalesce(title,'') || ' ' || coalesce(content,''))
            ) STORED
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_docs_teletext_teletext_id
    ON docs_teletext (teletext_id);

CREATE INDEX IF NOT EXISTS idx_docs_teletext_pub_datetime
    ON docs_teletext (publication_datetime);

CREATE INDEX IF NOT EXISTS docs_teletext_fts_gin
    ON docs_teletext
        USING gin (fts);

CREATE TABLE IF NOT EXISTS emb_teletext_chunk (
    id BIGSERIAL PRIMARY KEY,
    teletext_id TEXT NOT NULL REFERENCES docs_teletext (teletext_id),
    chunk_id INTEGER,
    chunk_text TEXT,
    embedding_em5 vector(1024)
);

CREATE INDEX IF NOT EXISTS idx_emb_teletext_chunk_embedding_em5_cosine
    ON emb_teletext_chunk
        USING ivfflat (embedding_em5 vector_cosine_ops)
    WITH (lists = 600); --numbers of clusters

CREATE UNIQUE INDEX IF NOT EXISTS idx_emb_teletext_chunk_id
    ON emb_teletext_chunk (teletext_id, chunk_id);

CREATE TABLE IF NOT EXISTS emb_teletext_full (
    id BIGSERIAL PRIMARY KEY,
    teletext_id TEXT NOT NULL REFERENCES docs_teletext (teletext_id),
    embedding_mpnet vector(768)
);

CREATE UNIQUE INDEX IF NOT EXISTS idx_emb_teletext_full_teletext_id
    ON emb_teletext_full (teletext_id);

CREATE INDEX IF NOT EXISTS idx_emb_teletext_full_embedding_mpnet_cosine
    ON emb_teletext_full
        USING ivfflat (embedding_mpnet vector_cosine_ops)
    WITH (lists = 600); --numbers of clusters
