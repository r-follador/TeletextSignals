from __future__ import annotations

from typing import List

from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever
import datetime
from modules.document_retriever import SearchResult, search_rerank


class TeletextHybridRetriever(BaseRetriever):
    """
    LangChain retriever for query_utils.search_rerank
    """

    k: int = 20
    top_k: int = 3
    score_floor: float = -1.0
    batch_size: int = 32
    debug: bool = False

    def _get_relevant_documents(self, query: str) -> List[Document]:
        results: List[SearchResult] = search_rerank(
            text_query=query,
            k=self.k,
            debug=self.debug,
            top_k=self.top_k,
            score_floor=self.score_floor,
            batch_size=self.batch_size,
        )

        docs: List[Document] = []
        for i, r in enumerate(results, start=1):
            # Prefer chunk text if present; otherwise use full article content
            text = r.get("content")

            dt = r.get("publication_datetime")
            dt_str = dt.date()

            docs.append(
                Document(
                    page_content=text,
                    metadata={
                        "rank": i,
                        "teletext_id": r.get("teletext_id"),
                        "chunk_id": r.get("chunk_id"),
                        "title": (r.get("title") or "").strip(),
                        "publication_datetime": dt_str,
                        "fts_score": r.get("fts_score"),
                        "cosine_similarity": r.get("cosine_similarity"),
                        "cross_score": r.get("cross_score"),
                    },
                )
            )

        return docs

def documents_to_search_results(docs: List[Document]) -> List[SearchResult]:
    """
    Convert LangChain Documents (from TeletextHybridRetriever)
    back into SearchResult dicts for notebook display.
    """
    out: List[SearchResult] = []

    for d in docs:
        md = d.metadata or {}

        pub = md.get("publication_datetime")
        try:
            publication_datetime = (
                datetime.datetime.fromisoformat(pub)
                if isinstance(pub, str)
                else pub
            )
        except Exception:
            publication_datetime = pub

        out.append(
            {
                "teletext_id": md.get("teletext_id") or "",
                "chunk_id": md.get("chunk_id"),
                "chunk_text": None,                 # not carried by this retriever
                "title": md.get("title") or "",
                "content": d.page_content or "",
                "publication_datetime": publication_datetime,
                "fts_score": md.get("fts_score"),
                "cosine_similarity": md.get("cosine_similarity"),
                "cross_score": md.get("cross_score"),
            }
        )

    return out