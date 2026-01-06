from IPython.display import display, HTML
from typing import Iterable, Optional
import re
from typing import Set
from typing import Iterable, List
from modules.query_utils import SearchResult

def display_search_results(
        rows: Iterable[SearchResult],
        *,
        max_content_chars: Optional[int] = None,
        citation_labels: Optional[dict[str, str]] = None,  # teletext_id -> label like "[1]"
):
    """
    Render search results as HTML cards in a Jupyter notebook.
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

        label = (citation_labels or {}).get(teletext_id)

        if max_content_chars is not None and content:
            content_display = content[:max_content_chars] + "…" if len(content) > max_content_chars else content
        else:
            content_display = content

        score_badges = []

        label_html = (
            f"""
            <span style="
                background-color: #fff3cd;
                color: #664d03;
                padding: 2px 6px;
                border-radius: 4px;
                margin-right: 6px;
                font-weight: 600;
                vertical-align: middle;
            ">{label}</span>
            """
            if label else ""
        )


        if fts_score is not None:
            score_badges.append(
                f"""
                <span style="background-color:#e6f2ff;color:#003366;padding:2px 6px;border-radius:4px;margin-right:6px;">
                    FTS: {fts_score:.4f}
                </span>
                """
            )

        if cosine_similarity is not None:
            score_badges.append(
                f"""
                <span style="background-color:#e8f8f0;color:#0a4d2f;padding:2px 6px;border-radius:4px;margin-right:6px;">
                    Vector: {cosine_similarity:.4f}
                </span>
                """
            )

        if cross_score is not None:
            score_badges.append(
                f"""
                <span style="background-color:#f3e8ff;color:#4b1b7a;padding:2px 6px;border-radius:4px;">
                    Cross: {cross_score:.4f}
                </span>
                """
            )

        scores_html = "".join(score_badges)

        pub_date = publication_datetime.date() if hasattr(publication_datetime, "date") else publication_datetime

        html = f"""
        <div style="border:1px solid #ddd;border-radius:6px;padding:12px;margin-bottom:12px;background-color:#fafafa;font-size:0.85rem;">
            <div style="display:flex;justify-content:space-between;align-items:baseline;">
                <h3 style="margin:0;font-family:ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace;font-size:0.95rem;font-weight:500;">
                    {label_html}{title} — {pub_date}
                </h3>
                <div style="font-size:0.8rem;">{scores_html}</div>
            </div>

            <p style="margin-top:8px;font-size:0.8rem;">
                <strong>Teletext ID</strong><br>{teletext_id}
                {f"<br><strong>Chunk ID</strong><br>{chunk_id}" if chunk_id is not None else ""}
            </p>

            <p style="font-size:0.8rem;">
                <strong>Content</strong><br>{content_display}
            </p>
        </div>
        """
        display(HTML(html))

def display_answer(answer):
    display(HTML(f"<div>{answer}</div>"))

_CIT_RE = re.compile(r"\[([0-9,\s]+)\]")

def extract_citation_indices(answer: str) -> list[int]: #extracts citations like [1],[2] and [3,4]
    nums: Set[int] = set()
    for m in _CIT_RE.finditer(answer or ""):
        for part in m.group(1).split(","):
            part = part.strip()
            if part.isdigit():
                n = int(part)
                if n > 0:
                    nums.add(n)
    return sorted(nums)

def display_cited_sources(answer: str, results: list[SearchResult], *, max_content_chars: int | None = 1200):
    cited = extract_citation_indices(answer)
    if not cited:
        print("No citations found in the answer.")
        return

    picked: list[SearchResult] = []
    labels: dict[str, str] = {}

    for idx in cited:
        pos = idx - 1
        if 0 <= pos < len(results):
            r = results[pos]
            picked.append(r)
            labels[r["teletext_id"]] = f"[{idx}]"

    if not picked:
        print("Citations were found, but none mapped to available results.")
        return

    display_search_results(picked, max_content_chars=max_content_chars, citation_labels=labels)


