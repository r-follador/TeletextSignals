import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder

DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "teletext"
DB_USER = "postgres"
DB_PASSWORD = "my-postgres-password"

# Same model as you used for ingestion
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

cross_encoder = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    device="cuda",
)


def embed_query(text: str) -> str:
    vec = embeddings.embed_query(f"query: {text}") # # add 'query: ' prefix, see https://huggingface.co/intfloat/multilingual-e5-large
    out_vec = "[" + ", ".join(map(str, vec)) + "]"
    return out_vec


def search_similar(text_query: str, k: int = 5):
    emb_str = embed_query(text_query)

    print(emb_str)

    conn = psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    cur = conn.cursor()

    sql = """
          SELECT
              r.teletext_id,
              r.chunk_id,
              r.chunk_text,
              t.title,
              t.content,
              1 - (r.embedding_em5 <=> %s::vector) AS cosine_similarity
          FROM emb_teletext_chunk r
          JOIN docs_teletext t
            ON t.teletext_id = r.teletext_id
          ORDER BY r.embedding_em5 <=> %s::vector
          LIMIT %s;
          """

    cur.execute(sql, (emb_str, emb_str, k))

    results = cur.fetchall()
    cur.close()
    conn.close()

    if not results:
        return []

    # Rerank the retrieved chunks with a cross-encoder for better precision
    # Use the full article context (title + content) for reranking.
    pairs = []
    for _, _, _, title, content, _ in results:
        doc_text = "\n\n".join(part for part in (title, content) if part)
        pairs.append((text_query, doc_text))

    cross_scores = cross_encoder.predict(pairs, batch_size=32)

    reranked = [
        (teletext_id, chunk_id, chunk, title, content, score, cross_score)
        for (teletext_id, chunk_id, chunk, title, content, score), cross_score in zip(results, cross_scores)
    ]

    reranked.sort(key=lambda x: x[6], reverse=True)
    return reranked


# --- Example usage ---
query = "Corona und Covid"
rows = search_similar(query, k=10)

for row in rows:
    teletext_id, chunk_id, chunk, title, content, vector_score, cross_score = row
    print(
        f"---Teletext: {teletext_id} | Chunk: {chunk_id} | "
        f"vector score: {vector_score:.4f} | cross score: {cross_score:.4f}"
    )
    print(f"Title: {title}")
    print(f"Content: {content}")
