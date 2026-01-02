"""
Embed Articles using mpnet https://huggingface.co/sentence-transformers/paraphrase-multilingual-mpnet-base-v2
- fulltext (implicitly limited to 512 tokens)
- 768 vector space
==> Used for clustering
"""

import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings

DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "teletext"
DB_USER = "postgres"
DB_PASSWORD = "my-postgres-password"


BATCH_SIZE = 100  # commit after this many articles

conn = psycopg2.connect(
    host=DB_HOST,
    port=DB_PORT,
    dbname=DB_NAME,
    user=DB_USER,
    password=DB_PASSWORD,
)
conn.autocommit = False
cur = conn.cursor()


def process_embeddings(rows):

    embeddings_mpnet = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={"device": "cuda"},
        encode_kwargs={"batch_size": 8, "normalize_embeddings": True},
    )

    articles_since_commit = 0

    for i, row in enumerate(rows):
        (
            teletext_row_id,
            teletext_id,
            title,
            content,
            publication_datetime,
            rubric,
            categories,
            language,
        ) = row

        if content is None:
            continue

        content = content.strip()
        if not content:
            continue

        title_str = (title or "").strip()
        full_text = (title_str + "\n\n" + content).strip()
        if not full_text:
            continue


        print(f"{i}/{len(rows)}: {publication_datetime}: {title} [{teletext_id}]")
        embedding = embeddings_mpnet.embed_documents([full_text])[0]


        # Turn embedding into a string representation understood by pgvector
        emb_768_str = "[" + ", ".join(map(str, embedding)) + "]"

        cur.execute(
            """
            INSERT INTO emb_teletext_full (
                teletext_id,
                embedding_mpnet
            )
            VALUES (%s, %s::vector)
                ON CONFLICT (teletext_id) DO UPDATE
                SET embedding_mpnet = EXCLUDED.embedding_mpnet
            """,
            (
                teletext_id,
                emb_768_str
            ),
        )

        articles_since_commit += 1

        if articles_since_commit >= BATCH_SIZE:
            print(f"-------Committing batch of {articles_since_commit} articles----------")
            conn.commit()
            articles_since_commit = 0

    conn.commit()


try:
    cur.execute(
        """
        SELECT id, teletext_id, title, content, publication_datetime, rubric, categories, language
        FROM docs_teletext t
        WHERE
            NOT EXISTS (
            SELECT 1
            FROM emb_teletext_full r
            WHERE r.teletext_id = t.teletext_id
            )
           OR
            EXISTS (
            SELECT 1
            FROM emb_teletext_full r
            WHERE r.teletext_id = t.teletext_id
          AND r.embedding_mpnet IS NULL
            );
        """
    )
    rows = cur.fetchall()

    process_embeddings(rows)

except Exception:
    conn.rollback()
    raise
finally:
    cur.close()
    conn.close()
