# Helper commands
## Postgres
### Recrate all indices

```sql
drop index idx_emb_teletext_chunk_embedding_em5_cosine;
SET maintenance_work_mem = '512MB';   -- more memory required for ivfflat index creation
CREATE INDEX IF NOT EXISTS idx_emb_teletext_chunk_embedding_em5_cosine
    ON emb_teletext_chunk
        USING ivfflat (embedding_em5 vector_cosine_ops)
    WITH (lists = 600); --numbers of clusters

drop index idx_emb_teletext_full_embedding_mpnet_cosine;
CREATE INDEX IF NOT EXISTS idx_emb_teletext_full_embedding_mpnet_cosine
    ON emb_teletext_full
        USING ivfflat (embedding_mpnet vector_cosine_ops)
    WITH (lists = 600); --numbers of clusters

drop index docs_teletext_fts_gin;
CREATE INDEX IF NOT EXISTS docs_teletext_fts_gin
    ON docs_teletext
        USING gin (fts);
```