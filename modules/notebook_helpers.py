from IPython.display import display, HTML
from typing import Iterable, Optional

def display_search_results(
        rows: Iterable[SearchResult],
        *,
        max_content_chars: Optional[int] = None,
):
    """
    Render search results as HTML cards in a Jupyter notebook.

    Highlights FTS score and/or cosine similarity depending on which
    modality contributed to the result.
    """
    for r in rows:
        teletext_id = r["teletext_id"]
        chunk_id = r.get("chunk_id")
        title = r.get("title") or ""
        content = r.get("content") or ""
        publication_datetime = r["publication_datetime"]
        fts_score = r.get("fts_score")
        cosine_similarity = r.get("cosine_similarity")
        cross_score = r.get("cross_score")

        if max_content_chars is not None and content:
            content_display = (
                content[:max_content_chars] + "…"
                if len(content) > max_content_chars
                else content
            )
        else:
            content_display = content

        # Build score badges (highlight source)
        score_badges = []

        if fts_score is not None:
            score_badges.append(
                f"""
                <span style="
                    background-color: #e6f2ff;
                    color: #003366;
                    padding: 2px 6px;
                    border-radius: 4px;
                    margin-right: 6px;
                ">
                    FTS: {fts_score:.4f}
                </span>
                """
            )

        if cosine_similarity is not None:
            score_badges.append(
                f"""
                <span style="
                    background-color: #e8f8f0;
                    color: #0a4d2f;
                    padding: 2px 6px;
                    border-radius: 4px;
                    margin-right: 6px;
                ">
                    Vector: {cosine_similarity:.4f}
                </span>
                """
            )

        if cross_score is not None:
            score_badges.append(
                f"""
                <span style="
                    background-color: #f3e8ff;
                    color: #4b1b7a;
                    padding: 2px 6px;
                    border-radius: 4px;
                ">
                    Cross: {cross_score:.4f}
                </span>
                """
            )

        scores_html = "".join(score_badges)

        pub_date = (
            publication_datetime.date()
            if hasattr(publication_datetime, "date")
            else publication_datetime
        )

        html = f"""
        <div style="
            border: 1px solid #ddd;
            border-radius: 6px;
            padding: 12px;
            margin-bottom: 12px;
            background-color: #fafafa;
            font-size: 0.85rem;
        ">
            <div style="display: flex; justify-content: space-between; align-items: baseline;">
                <h3 style="
                    margin: 0;
                    font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, 'Liberation Mono', monospace;
                    font-size: 0.95rem;
                    font-weight: 500;
                ">
                    {title} — {pub_date}
                </h3>
                <div style="font-size: 0.8rem;">
                    {scores_html}
                </div>
            </div>

            <p style="margin-top: 8px; font-size: 0.8rem;">
                <strong>Teletext ID</strong><br>
                {teletext_id}
                {f"<br><strong>Chunk ID</strong><br>{chunk_id}" if chunk_id is not None else ""}
            </p>

            <p style="font-size: 0.8rem;">
                <strong>Content</strong><br>
                {content_display}
            </p>
        </div>
        """

        display(HTML(html))
