from IPython.display import display, HTML, Markdown
from typing import TYPE_CHECKING, Any, Iterable, Optional
import re
from typing import Set
from typing import List

if TYPE_CHECKING:
    from modules.document_retriever import SearchResult
else:
    SearchResult = dict[str, Any]

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
            </p>

            <p style="font-size:0.8rem;">
                <strong>Content</strong><br>{content_display}
            </p>
        </div>
        """
        display(HTML(html))

def display_answer(answer):
    display(Markdown(
        f"""<div style='
        background: linear-gradient(180deg, #f0f9ff, #e0f2fe);
        border: 1px solid #bae6fd;
        border-radius: 10px;
        padding: 14px 16px;
        font-size: 1.05em;
        line-height: 1.55;
        box-shadow: 0 4px 12px rgba(0,0,0,0.06);'>\n{answer}\n</div>"""
    ))

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


def display_topic_timerange_results(
        topics: Iterable[dict[str, Any]],
        *,
        max_content_chars: Optional[int] = 800,
):
    """
    Render grouped topic results from get_top_topic_docs_for_timerange() in a notebook.
    """
    topics = list(topics)
    if not topics:
        display(HTML("<p><em>No topic documents found for this time range.</em></p>"))
        return

    for rank, topic in enumerate(topics, start=1):
        topic_id = topic.get("topic")
        topic_name = topic.get("topic_name") or "Unnamed topic"
        topic_representation = topic.get("topic_representation") or ""
        probability = topic.get("probability")
        rate_ratio = topic.get("rate_ratio")
        count_in_range = topic.get("count_in_range")
        count_before = topic.get("count_before")
        docs = topic.get("docs") or []

        chips = []
        if probability is not None:
            chips.append(f"<span style='background:#ecfeff;color:#155e75;padding:2px 8px;border-radius:999px;font-size:0.75rem;'>P={probability:.3f}</span>")
        if rate_ratio is not None:
            chips.append(f"<span style='background:#fef3c7;color:#92400e;padding:2px 8px;border-radius:999px;font-size:0.75rem;'>Rate x{rate_ratio:.2f}</span>")
        if count_in_range is not None and count_before is not None:
            chips.append(
                f"<span style='background:#e0e7ff;color:#3730a3;padding:2px 8px;border-radius:999px;font-size:0.75rem;'>Counts {count_in_range}/{count_before}</span>"
            )

        header_html = f"""
        <div style="margin:16px 0 8px 0;padding:12px 14px;border:1px solid #dbeafe;border-radius:10px;background:linear-gradient(180deg,#eff6ff,#f8fafc);">
            <div style="display:flex;justify-content:space-between;gap:12px;align-items:center;flex-wrap:wrap;">
                <div style="font-weight:700;color:#1e3a8a;">Top {rank}: Topic {topic_id} - {topic_name}</div>
                <div style="display:flex;gap:6px;flex-wrap:wrap;">{''.join(chips)}</div>
            </div>
            <div style="margin-top:6px;color:#334155;font-size:0.82rem;"><strong>Keywords:</strong> {topic_representation}</div>
            <div style="margin-top:6px;color:#475569;font-size:0.78rem;">{len(docs)} document(s) in selected range</div>
        </div>
        """
        display(HTML(header_html))
        display_search_results(docs, max_content_chars=max_content_chars)
