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

    cur.execute("SET ivfflat.probes = %s;", (60,)) # probes for ivfflat

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

    return results


# --- Example usage ---
query = "Wann gab es einen Brand in Trimmis GR?"
rows = search_similar(query, k=10)

for row in rows:
    teletext_id, chunk_id, chunk, title, content, vector_score = row
    print(
        f"---Teletext: {teletext_id} | Chunk: {chunk_id} | "
        f"vector score: {vector_score:.4f}"
    )
    print(f"Title: {title}")
    print(f"Content: {content}")
