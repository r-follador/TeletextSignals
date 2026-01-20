from typing import List, Optional, TypedDict

import psycopg2

DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "teletext"
DB_USER = "postgres"
DB_PASSWORD = "my-postgres-password"


class SearchResult(TypedDict):
    teletext_id: str
    chunk_id: Optional[int]
    chunk_text: Optional[str]
    title: str
    content: str
    publication_datetime: object
    fts_score: Optional[float]
    cosine_similarity: Optional[float]
    cross_score: Optional[float]


def mpnet_nearest_docs_by_teletext_id(
    teletext_id: str,
    *,
    k: int = 10,
    db_host: str = DB_HOST,
    db_port: str = DB_PORT,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_password: str = DB_PASSWORD,
    include_self: bool = False,
) -> List[SearchResult]:
    """
    Retrieve the closest k documents based on emb_teletext_full.embedding_mpnet.
    """
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    cur = conn.cursor()

    try:
        cur.execute("SET ivfflat.probes = %s;", (60,))

        sql = """
              WITH query_vec AS (
                  SELECT embedding_mpnet
                  FROM emb_teletext_full
                  WHERE teletext_id = %s
              )
              SELECT
                  t.teletext_id,
                  NULL AS chunk_id,
                  NULL AS chunk_text,
                  t.title,
                  t.content,
                  t.publication_datetime,
                  1 - (e.embedding_mpnet <=> q.embedding_mpnet) AS cosine_similarity
              FROM emb_teletext_full e
                       JOIN docs_teletext t
                            ON t.teletext_id = e.teletext_id
                       CROSS JOIN query_vec q
              ORDER BY e.embedding_mpnet <=> q.embedding_mpnet
                LIMIT %s;
        """

        cur.execute(sql, (teletext_id, k))
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    results: List[SearchResult] = []
    for teletext_id_row, chunk_id, chunk_text, title, content, publication_datetime, cosine_similarity in rows:
        results.append(
            {
                "teletext_id": teletext_id_row,
                "chunk_id": chunk_id,
                "chunk_text": chunk_text,
                "title": title,
                "content": content,
                "publication_datetime": publication_datetime,
                "fts_score": None,
                "cosine_similarity": cosine_similarity,
                "cross_score": None,
            }
        )

    return results
