import gc
import json
import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "teletext"
DB_USER = "postgres"
DB_PASSWORD = "my-postgres-password"

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=200,
    length_function=len,
)

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

    embeddings_e5 = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-large",
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

        chunks = text_splitter.split_text(full_text)
        if not chunks:
            continue

        print(f"{i}/{len(rows)}: {publication_datetime}: {title} [{teletext_id}]")
        chunk_embeddings = embeddings_e5.embed_documents(chunks)  # list[list[float]]

        for chunk_idx, (chunk_text, emb_vec) in enumerate(zip(chunks, chunk_embeddings)):

            # Turn embedding into a string representation understood by pgvector
            emb_1024_str = "[" + ", ".join(f"{x:.6f}" for x in emb_vec) + "]"

            cur.execute(
                """
                INSERT INTO emb_teletext_chunk (
                    teletext_id,
                    chunk_id,
                    chunk_text,
                    embedding_em5
                )
                VALUES (%s, %s, %s, %s::vector)
                    ON CONFLICT (teletext_id, chunk_id) DO NOTHING
                """,
                (
                    teletext_id,
                    chunk_idx,
                    chunk_text,
                    emb_1024_str
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
        WHERE NOT EXISTS (
            SELECT 1
            FROM emb_teletext_chunk r
            WHERE r.teletext_id = t.teletext_id
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
