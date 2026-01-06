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
    model_kwargs={"device": "cpu"}, # set this to 'cuda' if you have enough vram
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


class DocumentRetriever:
    def __init__(
        self,
        db_host: str = DB_HOST,
        db_port: str = DB_PORT,
        db_name: str = DB_NAME,
        db_user: str = DB_USER,
        db_password: str = DB_PASSWORD,
        embeddings_model: HuggingFaceEmbeddings = embeddings,
        cross_encoder_model: CrossEncoder = cross_encoder,
    ) -> None:
        self.db_host = db_host
        self.db_port = db_port
        self.db_name = db_name
        self.db_user = db_user
        self.db_password = db_password
        self.embeddings = embeddings_model
        self.cross_encoder = cross_encoder_model

    def embed_query(self, text: str) -> str:
        vec = self.embeddings.embed_query(
            f"query: {text}"
        )  # add 'query: ' prefix, see https://huggingface.co/intfloat/multilingual-e5-large
        out_vec = "[" + ", ".join(map(str, vec)) + "]"
        return out_vec

    def _get_conn(self):
        return psycopg2.connect(
            host=self.db_host,
            port=self.db_port,
            dbname=self.db_name,
            user=self.db_user,
            password=self.db_password,
        )

    def vector_similarity_search(
        self, text_query: str, k: int = 20, debug: bool = False
    ) -> List[SearchResult]:
        """
        Semantic (vector similarity) search using emb_teletext_chunk.embedding_em5 (vector).
        Returns normalized SearchResult dicts (TypedDict-compatible).
        """
        emb_str = self.embed_query(text_query)

        if debug:
            print(f"Vector search: k={k}, query={text_query!r}")
            #print(emb_str)

        conn = self._get_conn()
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
            conn.close()

    def fts_search(
        self,
        text_query: str,
        k: int = 20,
        debug: bool = False,
        language: str = "german",
    ) -> List[SearchResult]:
        """
        Full-text (lexical) search using docs_teletext.fts (tsvector)
        and websearch_to_tsquery. Returns normalized SearchResult dicts.
        """
        conn = self._get_conn()
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
            print(f"FTS search: language={language}, k={k}, query={text_query!r}")

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
        self,
        text_query: str,
        results: List[SearchResult],
        top_k: int = 10,
        score_floor: float = -0.5,
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

        cross_scores = self.cross_encoder.predict(pairs, batch_size=batch_size)

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
        reranked.sort(
            key=lambda x: x["cross_score"] if x["cross_score"] is not None else float("-inf"),
            reverse=True,
        )

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
        self,
        text_query: str,
        k: int = 20,
        debug: bool = False,
        top_k: int = 10,
        score_floor: float = -1.0,
        batch_size: int = 32,
    ) -> List[SearchResult]:
        results: List[SearchResult] = self.vector_similarity_search(text_query, k=k, debug=debug)
        return self.rerank_results(
            text_query=text_query,
            results=results,
            top_k=top_k,
            score_floor=score_floor,
            batch_size=batch_size,
        )

    def fts_search_rerank(
        self,
        text_query: str,
        k: int = 20,
        debug: bool = False,
        top_k: int = 10,
        score_floor: float = -1.0,
        batch_size: int = 32,
        language: str = "german",
    ) -> List[SearchResult]:
        results: List[SearchResult] = self.fts_search(text_query, k=k, debug=debug, language=language)
        return self.rerank_results(
            text_query=text_query,
            results=results,
            top_k=top_k,
            score_floor=score_floor,
            batch_size=batch_size,
        )

    def search_rerank(
        self,
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
        results_fts: List[SearchResult] = self.fts_search(
            text_query, k=k, debug=debug, language="german"
        )
        results_vec: List[SearchResult] = self.vector_similarity_search(
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
                f"After dedup: {len(deduped_results)}; "
                f"Returned results: top {top_k}"
            )

        return self.rerank_results(
            text_query=text_query,
            results=deduped_results,
            top_k=top_k,
            score_floor=score_floor,
            batch_size=batch_size,
        )


_default_retriever = DocumentRetriever()


def embed_query(text: str) -> str:
    return _default_retriever.embed_query(text)


def vector_similarity_search(text_query: str, k: int = 20, debug: bool = False) -> List[SearchResult]:
    return _default_retriever.vector_similarity_search(text_query, k=k, debug=debug)


def fts_search(
    text_query: str,
    k: int = 20,
    debug: bool = False,
    language: str = "german",
) -> List[SearchResult]:
    return _default_retriever.fts_search(text_query, k=k, debug=debug, language=language)


def rerank_results(
    text_query: str,
    results: List[SearchResult],
    top_k: int = 10,
    score_floor: float = -0.5,
    batch_size: int = 32,
) -> List[SearchResult]:
    return _default_retriever.rerank_results(
        text_query=text_query,
        results=results,
        top_k=top_k,
        score_floor=score_floor,
        batch_size=batch_size,
    )


def vector_similarity_rerank(
    text_query: str,
    k: int = 20,
    debug: bool = False,
    top_k: int = 10,
    score_floor: float = -1.0,
    batch_size: int = 32,
) -> List[SearchResult]:
    return _default_retriever.vector_similarity_rerank(
        text_query=text_query,
        k=k,
        debug=debug,
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
    return _default_retriever.fts_search_rerank(
        text_query=text_query,
        k=k,
        debug=debug,
        top_k=top_k,
        score_floor=score_floor,
        batch_size=batch_size,
        language=language,
    )


def search_rerank(
    text_query: str,
    k: int = 20,
    debug: bool = False,
    top_k: int = 10,
    score_floor: float = -1.0,
    batch_size: int = 32,
) -> List[SearchResult]:
    return _default_retriever.search_rerank(
        text_query=text_query,
        k=k,
        debug=debug,
        top_k=top_k,
        score_floor=score_floor,
        batch_size=batch_size,
    )
