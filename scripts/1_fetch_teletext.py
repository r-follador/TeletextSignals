#!/usr/bin/env python3
"""
Fetch Swiss Teletext articles and upsert them into docs_teletext.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from typing import Any, Dict, Iterable, List, Optional

import psycopg
import requests


BASE_URL = "https://api.teletext.ch/webtext/Articles/search/de"

DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "teletext"
DB_USER = "postgres"
DB_PASSWORD = "my-postgres-password"


def db_connect() -> psycopg.Connection:
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    conn.autocommit = False
    return conn


def fetch_page(page: int) -> Dict[str, Any] | List[Any]:
    print(f"-- Fetch page {page} ---")
    retries = 5
    for attempt in range(retries + 1):
        resp = requests.get(BASE_URL, params={"page": page}, timeout=30)
        try:
            resp.raise_for_status()
        except requests.HTTPError:
            if resp.status_code == 502 and attempt < retries:
                print(f"502 from teletext API, retrying in 30s ({attempt + 1}/{retries})...")
                time.sleep(30)
                continue
            raise
        return resp.json()
    raise RuntimeError("Unreachable")


def _extract_items(payload: Dict[str, Any] | List[Any]) -> List[Dict[str, Any]]:
    if isinstance(payload, list):
        return [item for item in payload if isinstance(item, dict)]
    if isinstance(payload, dict):
        for key in ("articles", "data", "items", "results", "content"):
            value = payload.get(key)
            if isinstance(value, list):
                return [item for item in value if isinstance(item, dict)]
    return []


def _extract_total_pages(payload: Dict[str, Any] | List[Any]) -> Optional[int]:
    if not isinstance(payload, dict):
        return None
    for key in ("pageCount", "totalPages", "pages", "lastPage", "total_pages", "page_count"):
        value = payload.get(key)
        if isinstance(value, int):
            return value
    pagination = payload.get("pagination")
    if isinstance(pagination, dict):
        for key in ("totalPages", "pages", "lastPage"):
            value = pagination.get(key)
            if isinstance(value, int):
                return value
    return None


def _normalize_text(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        cleaned = value.strip()
        return cleaned or None
    if isinstance(value, list):
        parts: List[str] = []
        for item in value:
            if isinstance(item, str):
                parts.append(item.strip())
            elif isinstance(item, dict):
                for key in ("text", "content", "value"):
                    if key in item and isinstance(item[key], str):
                        parts.append(item[key].strip())
                        break
                else:
                    parts.append(json.dumps(item, ensure_ascii=True))
            else:
                parts.append(str(item))
        joined = "\n".join([p for p in parts if p])
        return joined or None
    if isinstance(value, dict):
        for key in ("text", "content", "value"):
            if key in value and isinstance(value[key], str):
                cleaned = value[key].strip()
                return cleaned or None
        return json.dumps(value, ensure_ascii=True)
    return str(value)


def _normalize_categories(value: Any) -> Optional[List[str]]:
    if value is None:
        return None
    if isinstance(value, str):
        stripped = value.strip()
        if stripped.startswith("[") and stripped.endswith("]"):
            try:
                parsed = json.loads(stripped)
                if isinstance(parsed, list):
                    value = parsed
            except json.JSONDecodeError:
                pass
    categories: List[str] = []
    if isinstance(value, list):
        items = value
    else:
        items = [value]
    for item in items:
        if isinstance(item, str):
            categories.append(item)
        elif isinstance(item, dict):
            for key in ("name", "title", "label", "id"):
                if key in item and item[key] is not None:
                    categories.append(str(item[key]))
                    break
        else:
            categories.append(str(item))
    categories = [c.strip() for c in categories if c and c.strip()]
    return categories or None


def _parse_datetime(value: Any) -> Optional[dt.datetime]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        ts = float(value)
        if ts > 1_000_000_000_000:
            ts = ts / 1000.0
        return dt.datetime.fromtimestamp(ts, tz=dt.timezone.utc)
    if isinstance(value, str):
        cleaned = value.strip()
        if not cleaned:
            return None
        if cleaned.endswith("Z"):
            cleaned = cleaned[:-1] + "+00:00"
        try:
            parsed = dt.datetime.fromisoformat(cleaned)
            if parsed.tzinfo is None:
                return parsed.replace(tzinfo=dt.timezone.utc)
            return parsed
        except ValueError:
            return None
    return None


def parse_article(article: Dict[str, Any]) -> Dict[str, Any]:
    teletext_id = (
        article.get("Id")
        or article.get("id")
        or article.get("articleId")
        or article.get("article_id")
        or article.get("key")
    )
    title = article.get("Title") or article.get("title") or article.get("headline") or article.get("teaser")
    content = _normalize_text(
        article.get("TextContent")
        or article.get("text")
        or article.get("content")
        or article.get("body")
        or article.get("textContent")
        or article.get("articleText")
    )
    publication_datetime = _parse_datetime(
        article.get("PublicationDateTime")
        or article.get("publicationDate")
        or article.get("publishedAt")
        or article.get("date")
        or article.get("pubDate")
        or article.get("timestamp")
    )
    rubric = (
        article.get("Rubric")
        or article.get("SourceRubric")
        or article.get("rubric")
        or article.get("section")
        or article.get("category")
    )
    categories = _normalize_categories(
        article.get("Categories") or article.get("categories") or article.get("topics")
    )
    language = article.get("Language") or article.get("language") or article.get("lang") or article.get("locale")

    return {
        "teletext_id": str(teletext_id) if teletext_id is not None else None,
        "title": title,
        "content": content,
        "publication_datetime": publication_datetime,
        "rubric": rubric,
        "categories": categories,
        "language": language,
    }


def fetch_all(start_page, max_pages: Optional[int]) -> Iterable[Dict[str, Any]]:
    page = start_page

    while True:
        payload = fetch_page(page)
        items = _extract_items(payload)
        if not items:
            break

        for item in items:
            doc = parse_article(item)
            if doc["teletext_id"] is None:
                continue
            _log_article(doc)
            yield doc

        total_pages = _extract_total_pages(payload)
        if total_pages is not None and page >= total_pages:
            break

        page += 1
        if max_pages is not None and page > max_pages:
            break


def chunked(seq: List[Dict[str, Any]], size: int) -> Iterable[List[Dict[str, Any]]]:
    for i in range(0, len(seq), size):
        yield seq[i : i + size]


def _log_article(doc: Dict[str, Any]) -> None:
    title = doc.get("title") or "<untitled>"
    pub_dt = doc.get("publication_datetime")
    if isinstance(pub_dt, dt.datetime):
        timestamp = pub_dt.isoformat(timespec="seconds")
    else:
        timestamp = "unknown date"
    print(f"{timestamp} | {title}")


def upsert_documents(conn: psycopg.Connection, docs: Iterable[Dict[str, Any]]) -> None:
    rows = list(docs)
    if not rows:
        return
    for batch in chunked(rows, 100):
        with conn.cursor() as cur:
            cur.executemany(
                """
                INSERT INTO docs_teletext (
                    teletext_id,
                    title,
                    content,
                    publication_datetime,
                    rubric,
                    categories,
                    language
                )
                VALUES (
                    %(teletext_id)s,
                    %(title)s,
                    %(content)s,
                    %(publication_datetime)s,
                    %(rubric)s,
                    %(categories)s,
                    %(language)s
                )
                ON CONFLICT (teletext_id) DO UPDATE
                SET
                    title = EXCLUDED.title,
                    content = EXCLUDED.content,
                    publication_datetime = EXCLUDED.publication_datetime,
                    rubric = EXCLUDED.rubric,
                    categories = EXCLUDED.categories,
                    language = EXCLUDED.language
                """,
                batch,
            )
        conn.commit()


def run(start_page, max_pages: Optional[int]) -> int:
    docs = fetch_all(start_page, max_pages)
    buffer: List[Dict[str, Any]] = []
    total = 0

    with db_connect() as conn:
        for doc in docs:
            buffer.append(doc)
            total += 1
            if len(buffer) >= 100:
                upsert_documents(conn, buffer)
                buffer.clear()

        if buffer:
            upsert_documents(conn, buffer)

    if total == 0:
        print("No teletext articles found.")
        return 0

    print(f"Upserted {total} teletext articles into docs_teletext.")
    return 0



run(1,60120) #earliest is 2000-04-10 (https://api.teletext.ch/webtext/Articles/search/de?&page=60109)
