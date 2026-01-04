import datetime

import psycopg2
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import CrossEncoder
from typing import TypedDict, Optional, List

DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "teletext"
DB_USER = "postgres"
DB_PASSWORD = "my-postgres-password"

# Same model as used for ingestion
# see https://huggingface.co/intfloat/multilingual-e5-large
embeddings = HuggingFaceEmbeddings(
    model_name="intfloat/multilingual-e5-large",
    model_kwargs={"device": "cuda"},
    encode_kwargs={"normalize_embeddings": True},
)

# Cross encoder for reranking
# see https://huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
cross_encoder = CrossEncoder(
    "cross-encoder/mmarco-mMiniLMv2-L12-H384-v1",
    device="cuda",
)

class SearchResult(TypedDict):
    teletext_id: str
    chunk_id: Optional[int]
    chunk_text: Optional[str]
    title: str
    content: str
    publication_datetime: datetime.datetime
    fts_score: Optional[float]
    cosine_similarity: Optional[float]
    cross_score: Optional[float]


def embed_query(text: str) -> str:
    vec = embeddings.embed_query(
        f"query: {text}"
    )  # add 'query: ' prefix, see https://huggingface.co/intfloat/multilingual-e5-large
    out_vec = "[" + ", ".join(map(str, vec)) + "]"
    return out_vec


def _get_conn():
    return psycopg2.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )



def vector_similarity_search(text_query: str, k: int = 20, debug: bool = False) -> List[SearchResult]:
    """
    Semantic (vector similarity) search using emb_teletext_chunk.embedding_em5 (vector).
    Returns normalized SearchResult dicts (TypedDict-compatible).
    """
    emb_str = embed_query(text_query)

    if debug:
        print(emb_str)

    conn = _get_conn()
    cur = conn.cursor()

    try:
        cur.execute("SET ivfflat.probes = %s;", (60,))  # probes for ivfflat

        sql = """
              SELECT
                  r.teletext_id,
                  r.chunk_id,
                  r.chunk_text,
                  t.title,
                  t.content,
                  t.publication_datetime,
                  1 - (r.embedding_em5 <=> %s::vector) AS cosine_similarity
              FROM emb_teletext_chunk r
                       JOIN docs_teletext t
                            ON t.teletext_id = r.teletext_id
              ORDER BY r.embedding_em5 <=> %s::vector
                  LIMIT %s;
              """

        cur.execute(sql, (emb_str, emb_str, k))
        rows = cur.fetchall()

        results: List[SearchResult] = []
        for teletext_id, chunk_id, chunk_text, title, content, publication_datetime, cosine_similarity in rows:
            results.append(
                {
                    "teletext_id": teletext_id,
                    "chunk_id": chunk_id,
                    "chunk_text": chunk_text,
                    "title": title,
                    "content": content,
                    "publication_datetime": publication_datetime,
                    "fts_score": None,
                    "cosine_similarity": float(cosine_similarity) if cosine_similarity is not None else None,
                    "cross_score": None,
                }
            )

        return results

    finally:
        cur.close()


def fts_search(
        text_query: str,
        k: int = 20,
        debug: bool = False,
        language: str = "german",
) -> List[SearchResult]:
    """
    Full-text (lexical) search using docs_teletext.fts (tsvector)
    and websearch_to_tsquery. Returns normalized SearchResult dicts.
    """
    conn = _get_conn()
    cur = conn.cursor()

    sql = """
          SELECT
              t.teletext_id,
              NULL::integer AS chunk_id,
              NULL::text    AS chunk_text,
              t.title,
              t.content,
              t.publication_datetime,
              ts_rank_cd(t.fts, q) AS fts_score
          FROM docs_teletext t,
               websearch_to_tsquery(%s, %s) AS q
          WHERE t.fts @@ q
          ORDER BY fts_score DESC
              LIMIT %s;
          """

    if debug:
        print(f"FTS language={language}, k={k}, query={text_query!r}")

    try:
        cur.execute(sql, (language, text_query, k))
        rows = cur.fetchall()

        results: List[SearchResult] = []
        for teletext_id, chunk_id, chunk_text, title, content, publication_datetime, fts_score in rows:
            results.append(
                {
                    "teletext_id": teletext_id,
                    "chunk_id": None,
                    "chunk_text": None,
                    "title": title,
                    "content": content,
                    "publication_datetime": publication_datetime,
                    "fts_score": float(fts_score) if fts_score is not None else None,
                    "cosine_similarity": None,
                    "cross_score": None,
                }
            )

        return results

    finally:
        cur.close()
        conn.close()


def rerank_results(
        text_query: str,
        results: List[SearchResult],
        top_k: int = 10,
        score_floor: float = -1.0,
        batch_size: int = 32,
) -> List[SearchResult]:
    """
    Cross-encoder rerank using full article context (title + content).
    Expects normalized SearchResult dicts and sets `cross_score`.
    """
    if not results:
        return []

    # Build (query, doc) pairs for the cross-encoder.
    pairs: list[tuple[str, str]] = []
    for r in results:
        title = r.get("title") or ""
        content = r.get("content") or ""
        doc_text = "\n\n".join(part for part in (title, content) if part)
        pairs.append((text_query, doc_text))

    cross_scores = cross_encoder.predict(pairs, batch_size=batch_size)

    reranked: List[SearchResult] = []
    for r, cs in zip(results, cross_scores):
        rr: SearchResult = {
            "teletext_id": r["teletext_id"],
            "chunk_id": r.get("chunk_id"),
            "chunk_text": r.get("chunk_text"),
            "title": r["title"],
            "content": r["content"],
            "publication_datetime": r["publication_datetime"],
            "fts_score": r.get("fts_score"),
            "cosine_similarity": r.get("cosine_similarity"),
            "cross_score": float(cs) if cs is not None else None,
        }
        reranked.append(rr)

    # Sort by cross_score desc; treat None as very low.
    reranked.sort(key=lambda x: x["cross_score"] if x["cross_score"] is not None else float("-inf"), reverse=True)

    # Apply top_k and score floor (exclude None or below floor)
    out: List[SearchResult] = []
    for r in reranked:
        cs = r["cross_score"]
        if cs is None or cs <= score_floor:
            continue
        out.append(r)
        if len(out) >= top_k:
            break

    return out


def vector_similarity_rerank(
        text_query: str,
        k: int = 20,
        debug: bool = False,
        top_k: int = 10,
        score_floor: float = -1.0,
        batch_size: int = 32,
) -> List[SearchResult]:
    results: List[SearchResult] = vector_similarity_search(text_query, k=k, debug=debug)
    return rerank_results(
        text_query=text_query,
        results=results,
        top_k=top_k,
        score_floor=score_floor,
        batch_size=batch_size,
    )

def fts_search_rerank(
        text_query: str,
        k: int = 20,
        debug: bool = False,
        top_k: int = 10,
        score_floor: float = -1.0,
        batch_size: int = 32,
        language: str = "german",
) -> List[SearchResult]:
    results: List[SearchResult] = fts_search(text_query, k=k, debug=debug, language=language)
    return rerank_results(
        text_query=text_query,
        results=results,
        top_k=top_k,
        score_floor=score_floor,
        batch_size=batch_size,
    )


def search_rerank(
        text_query: str,
        k: int = 20,
        debug: bool = False,
        top_k: int = 10,
        score_floor: float = -1.0,
        batch_size: int = 32,
) -> List[SearchResult]:
    """
    Run FTS and vector similarity search, deduplicate by teletext_id,
    preserve both score types, then cross-encoder rerank.
    """
    results_fts: List[SearchResult] = fts_search(
        text_query, k=k, debug=debug, language="german"
    )
    results_vec: List[SearchResult] = vector_similarity_search(
        text_query, k=k, debug=debug
    )

    # Merge and deduplicate by teletext_id, preserving both scores
    merged: dict[str, SearchResult] = {}

    for r in results_fts + results_vec:
        teletext_id = r["teletext_id"]

        if teletext_id not in merged:
            merged[teletext_id] = r
        else:
            m = merged[teletext_id]

            # Prefer chunk-level info if present (vector search)
            if m.get("chunk_id") is None and r.get("chunk_id") is not None:
                m["chunk_id"] = r["chunk_id"]
                m["chunk_text"] = r["chunk_text"]

            # Merge scores (do NOT compare across modalities)
            if m.get("fts_score") is None and r.get("fts_score") is not None:
                m["fts_score"] = r["fts_score"]

            if m.get("cosine_similarity") is None and r.get("cosine_similarity") is not None:
                m["cosine_similarity"] = r["cosine_similarity"]

            merged[teletext_id] = m

    deduped_results = list(merged.values())

    if debug:
        print(
            f"FTS results: {len(results_fts)}, "
            f"Vector results: {len(results_vec)}, "
            f"After dedup: {len(deduped_results)}"
        )

    return rerank_results(
        text_query=text_query,
        results=deduped_results,
        top_k=top_k,
        score_floor=score_floor,
        batch_size=batch_size,
    )