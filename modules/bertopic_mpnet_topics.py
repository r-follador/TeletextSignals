import ast
import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import psycopg2
from psycopg2.extras import execute_values
from tqdm import tqdm

from bertopic import BERTopic
from hdbscan import HDBSCAN

DB_HOST = "localhost"
DB_PORT = "5433"
DB_NAME = "teletext"
DB_USER = "postgres"
DB_PASSWORD = "my-postgres-password"
TOPIC_INFO_TABLE = "bertopic_topic_info"


def _parse_embedding(value: Any) -> Optional[np.ndarray]:
    if value is None:
        return None
    if isinstance(value, (list, tuple)):
        if not value:
            return None
        return np.asarray(value, dtype=np.float32)
    if isinstance(value, np.ndarray):
        if value.size == 0:
            return None
        return value.astype(np.float32, copy=False)
    if isinstance(value, str):
        text = value.strip()
        if not text:
            return None
        # Fast path for pgvector-like string representations: "[0.1, 0.2, ...]"
        if (text[0] == "[" and text[-1] == "]") or (text[0] == "{" and text[-1] == "}"):
            parsed = np.fromstring(text[1:-1], sep=",", dtype=np.float32)
            if parsed.size:
                return parsed
        try:
            parsed = ast.literal_eval(text)
        except (ValueError, SyntaxError):
            return None
        if isinstance(parsed, (list, tuple)) and parsed:
            return np.asarray(parsed, dtype=np.float32)
    return None


def load_mpnet_documents(
    *,
    db_host: str = DB_HOST,
    db_port: str = DB_PORT,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_password: str = DB_PASSWORD,
) -> Tuple[List[str], List[pd.Timestamp], np.ndarray, List[str]]:
    print("Loading teletext documents and MPNet embeddings...")
    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    cur = conn.cursor()

    try:
        cur.execute(
            """
            SELECT
                d.teletext_id,
                d.title,
                d.content,
                d.publication_datetime,
                e.embedding_mpnet
            FROM docs_teletext d
            JOIN emb_teletext_full e ON e.teletext_id = d.teletext_id
            WHERE e.embedding_mpnet IS NOT NULL
              AND d.publication_datetime IS NOT NULL
              AND (rubric IS NULL OR rubric NOT LIKE 'Sport%'); --exclude sport news, not interested
            --AND d.publication_datetime >= '2025-01-01 00:00:00';
            """
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    print(f"Fetched {len(rows)} rows from the database.")
    docs: List[str] = []
    timestamps_raw: List[Any] = []
    embeddings: List[np.ndarray] = []
    teletext_ids: List[str] = []

    total_rows = len(rows)
    for teletext_id, title, content, publication_datetime, embedding_value in tqdm(
        rows,
        total=total_rows,
        desc="Parsing rows",
        unit="row",
    ):
        title_str = (title or "").strip()
        content_str = (content or "").strip()
        full_text = (title_str + "\n\n" + content_str).strip()
        if not full_text:
            continue

        embedding = _parse_embedding(embedding_value)
        if embedding is None or embedding.size == 0:
            continue

        docs.append(full_text)
        embeddings.append(embedding)
        teletext_ids.append(teletext_id)
        timestamps_raw.append(publication_datetime)

    if not embeddings:
        raise ValueError("No embeddings found for BERTopic processing.")

    timestamps = list(pd.to_datetime(timestamps_raw))
    print(
        "Prepared "
        f"{len(docs)} documents, "
        f"{len(teletext_ids)} teletext ids, "
        f"{len(embeddings)} embeddings."
    )

    #L2 normalization
    emb = np.asarray(embeddings, dtype=np.float32)
    emb /= np.linalg.norm(emb, axis=1, keepdims=True) + 1e-12

    return docs, timestamps, emb, teletext_ids


def build_topic_model() -> BERTopic:
    # HDBSCAN defaults: min_cluster_size=5, min_samples=None, metric="euclidean", cluster_selection_method="eom"
    hdbscan_model = HDBSCAN(
        min_cluster_size=20,
        min_samples=5,
        cluster_selection_method="leaf",
        # Use euclidean with L2-normalized embeddings (cosine-equivalent) to
        # avoid sklearn/hdbscan metric incompatibility in BallTree.
        metric="euclidean",
        prediction_data=False
    )
    # BERTopic defaults: top_n_words=10, min_topic_size=10, n_gram_range=(1, 1), verbose=False
    return BERTopic(
        embedding_model=None,
        umap_model=None,
        hdbscan_model=hdbscan_model,
        verbose=True,
    )


def _build_topic_teletext_map(
    topics: Sequence[int],
    teletext_ids: Sequence[str],
) -> Dict[int, List[str]]:
    topic_map: Dict[int, List[str]] = {}
    for topic_id, teletext_id in zip(topics, teletext_ids):
        topic_map.setdefault(int(topic_id), []).append(teletext_id)
    return topic_map


def store_topic_info(
    *,
    topic_model: BERTopic,
    topics: Sequence[int],
    teletext_ids: Sequence[str],
    db_host: str = DB_HOST,
    db_port: str = DB_PORT,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_password: str = DB_PASSWORD,
    table_name: str = TOPIC_INFO_TABLE,
) -> None:
    topic_info = topic_model.get_topic_info()
    topic_map = _build_topic_teletext_map(topics, teletext_ids)
    rows: List[Tuple[int, int, Optional[str], Optional[List[str]], List[str]]] = []

    for _, row in topic_info.iterrows():
        topic_id = int(row["Topic"])
        topic_count = int(row["Count"])
        topic_name = row.get("Name")
        representation = row.get("Representation")
        if isinstance(representation, str):
            representation = [representation]
        if representation is not None:
            representation = [str(item) for item in representation]
        rows.append(
            (
                topic_id,
                topic_count,
                topic_name if pd.notna(topic_name) else None,
                representation if representation is not None else None,
                topic_map.get(topic_id, []),
            )
        )

    if not rows:
        return

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    cur = conn.cursor()
    try:
        execute_values(
            cur,
            f"""
            INSERT INTO {table_name}
                (topic, topic_count, topic_name, topic_representation, teletext_ids)
            VALUES %s
            """,
            rows,
        )
        conn.commit()
    finally:
        cur.close()
        conn.close()

def run_topics(
    *,
    db_host: str = DB_HOST,
    db_port: str = DB_PORT,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_password: str = DB_PASSWORD,
) -> Tuple[BERTopic, dict]:
    print("Starting BERTopic topic pipeline...") # excludes sports news
    docs, _, embeddings, teletext_ids = load_mpnet_documents(
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
    )

    print("Building BERTopic model (HDBSCAN)...")
    topic_model = build_topic_model()
    print("Fitting BERTopic model with existing embeddings...")
    topics, _ = topic_model.fit_transform(docs, embeddings)
    print("Storing BERTopic topic info in database...")
    store_topic_info(
        topic_model=topic_model,
        topics=topics,
        teletext_ids=teletext_ids,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
    )
    summary = {
        "documents": len(docs),
        "unique_topics": len({topic for topic in topics if topic != -1}),
        "outliers": sum(1 for topic in topics if topic == -1),
        "teletext_ids": len(set(teletext_ids)),
    }
    return topic_model, summary


def _posterior_probability(
    *,
    count_range: int,
    exposure_range: int,
    count_before: int,
    exposure_before: int,
    alpha: float,
    beta: float,
    delta: float,
    draws: int,
    rng: np.random.Generator,
) -> Tuple[float, float, float, float, float]:
    if exposure_range <= 0 or exposure_before <= 0:
        raise ValueError("Exposure must be positive for rate modeling.")
    if alpha <= 0 or beta <= 0:
        raise ValueError("alpha and beta must be positive for Gamma prior.")

    alpha_range = alpha + count_range
    beta_range = beta + exposure_range
    alpha_before = alpha + count_before
    beta_before = beta + exposure_before

    lambda_range = rng.gamma(shape=alpha_range, scale=1.0 / beta_range, size=draws)
    lambda_before = rng.gamma(shape=alpha_before, scale=1.0 / beta_before, size=draws)
    prob = float(np.mean(lambda_range > (lambda_before + delta)))

    mean_range = alpha_range / beta_range
    mean_before = alpha_before / beta_before
    ratio = mean_range / mean_before if mean_before > 0 else math.inf
    mean_diff = float(np.mean(lambda_range - lambda_before))
    return prob, mean_range, mean_before, ratio, mean_diff


def find_significant_topics(
    *,
    timerange_start: str,
    timerange_end: str,
    min_topic_count: int = 5,
    alpha_prior: float = 0.5,
    beta_prior: float = 1.0,
    delta: float = 0.0,
    draws: int = 20000,
    min_probability: float = 0.9,
    random_seed: int = 7,
    db_host: str = DB_HOST,
    db_port: str = DB_PORT,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_password: str = DB_PASSWORD,
    table_name: str = TOPIC_INFO_TABLE,
) -> pd.DataFrame:
    """Rank topic rate shifts using Gamma-Poisson posteriors."""
    start_ts = pd.to_datetime(timerange_start)
    end_ts = pd.to_datetime(timerange_end)
    if end_ts <= start_ts:
        raise ValueError("timerange_end must be after timerange_start")
    end_exclusive = end_ts + pd.Timedelta(days=1)

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    cur = conn.cursor()
    try:
        cur.execute(
            """
            SELECT
                COUNT(*) FILTER (
                    WHERE publication_datetime >= %(start)s
                      AND publication_datetime < %(end)s
                ) AS total_range,
                COUNT(*) FILTER (
                    WHERE publication_datetime < %(start)s
                ) AS total_before
            FROM docs_teletext
            WHERE publication_datetime IS NOT NULL
            """,
            {"start": start_ts, "end": end_exclusive},
        )
        total_range, total_before = cur.fetchone()

        if total_range == 0 or total_before == 0:
            raise ValueError("Not enough documents in the selected range or baseline.")

        cur.execute(
            f"""
            WITH topic_docs AS (
                SELECT DISTINCT
                    b.topic,
                    b.topic_name,
                    b.topic_representation,
                    unnest(b.teletext_ids) AS teletext_id
                FROM {table_name} b
            )
            SELECT
                t.topic,
                t.topic_name,
                t.topic_representation,
                COUNT(*) FILTER (
                    WHERE d.publication_datetime >= %(start)s
                      AND d.publication_datetime < %(end)s
                ) AS count_in_range,
                COUNT(*) FILTER (
                    WHERE d.publication_datetime < %(start)s
                ) AS count_before
            FROM topic_docs t
            JOIN docs_teletext d ON d.teletext_id = t.teletext_id
            WHERE d.publication_datetime IS NOT NULL
            GROUP BY t.topic, t.topic_name, t.topic_representation
            ORDER BY t.topic
            """,
            {"start": start_ts, "end": end_exclusive},
        )
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    if not rows:
        raise ValueError("No topic rows returned from bertopic_topic_info.")

    df = pd.DataFrame(
        rows,
        columns=[
            "topic",
            "topic_name",
            "topic_representation",
            "count_in_range",
            "count_before",
        ],
    )

    df = df[df["topic"] != -1].copy()
    df["total_count"] = df["count_in_range"] + df["count_before"]
    df = df[df["total_count"] >= min_topic_count].copy()

    probs: List[float] = []
    mean_range_list: List[float] = []
    mean_before_list: List[float] = []
    ratio_list: List[float] = []
    diff_list: List[float] = []
    rng = np.random.default_rng(random_seed)
    for _, row in df.iterrows():
        prob, mean_range, mean_before, ratio, mean_diff = _posterior_probability(
            count_range=int(row["count_in_range"]),
            exposure_range=int(total_range),
            count_before=int(row["count_before"]),
            exposure_before=int(total_before),
            alpha=alpha_prior,
            beta=beta_prior,
            delta=delta,
            draws=draws,
            rng=rng,
        )
        probs.append(prob)
        mean_range_list.append(mean_range)
        mean_before_list.append(mean_before)
        ratio_list.append(ratio)
        diff_list.append(mean_diff)

    df["rate_in_range"] = mean_range_list
    df["rate_before"] = mean_before_list
    df["rate_ratio"] = ratio_list
    df["mean_diff"] = diff_list
    df["probability"] = probs
    df["direction"] = np.where(df["rate_ratio"] >= 1.0, "up", "down")

    df = df.sort_values("rate_ratio", ascending=False)
    significant = df[df["probability"] >= min_probability].copy()

    print(
        "Topics with P(lambda1 > lambda0 + delta) >= {min_prob} "
        "for {start} to {end} vs prior history:".format(
            start=start_ts.date(), end=end_ts.date(), min_prob=min_probability
        )
    )
    if significant.empty:
        print("No significant topic deviations found.")
        return df

    def _repr_text(value: Any) -> str:
        if value is None or (isinstance(value, float) and math.isnan(value)):
            return ""
        if isinstance(value, list):
            return ", ".join(str(item) for item in value)
        return str(value)

    output = significant.copy()
    output["topic_representation"] = output["topic_representation"].apply(_repr_text)
    output["rate_ratio"] = output["rate_ratio"].round(3)
    output["probability"] = output["probability"].map(lambda v: f"{v:.3f}")
    output["mean_diff"] = output["mean_diff"].round(6)

    """
    print(
        output[
            [
                "topic",
                "topic_name",
                "direction",
                "count_in_range",
                "count_before",
                "rate_ratio",
                "mean_diff",
                "probability",
                "topic_representation",
            ]
        ].to_string(index=False)
    )
    """
    return df


def get_top_topic_docs_for_timerange(
    *,
    timerange_start: str,
    timerange_end: str,
    min_topic_count: int = 5,
    alpha_prior: float = 0.5,
    beta_prior: float = 1.0,
    delta: float = 0.0,
    draws: int = 20000,
    min_probability: float = 0.9,
    random_seed: int = 7,
    limit: Optional[int] = None,
    db_host: str = DB_HOST,
    db_port: str = DB_PORT,
    db_name: str = DB_NAME,
    db_user: str = DB_USER,
    db_password: str = DB_PASSWORD,
    table_name: str = TOPIC_INFO_TABLE,
) -> List[Dict[str, Any]]:
    """Return docs for the top topic in a timerange."""
    df = find_significant_topics(
        timerange_start=timerange_start,
        timerange_end=timerange_end,
        min_topic_count=min_topic_count,
        alpha_prior=alpha_prior,
        beta_prior=beta_prior,
        delta=delta,
        draws=draws,
        min_probability=min_probability,
        random_seed=random_seed,
        db_host=db_host,
        db_port=db_port,
        db_name=db_name,
        db_user=db_user,
        db_password=db_password,
        table_name=table_name,
    )

    if df.empty:
        print("No topics available for the requested time range.")
        return []

    eligible = df[df["probability"] >= min_probability] if "probability" in df.columns else df
    if eligible.empty:
        print("No topics passed the probability threshold; using top ranked topic anyway.")
        eligible = df

    top_topic = int(eligible.iloc[0]["topic"])

    start_ts = pd.to_datetime(timerange_start)
    end_ts = pd.to_datetime(timerange_end)

    conn = psycopg2.connect(
        host=db_host,
        port=db_port,
        dbname=db_name,
        user=db_user,
        password=db_password,
    )
    cur = conn.cursor()
    try:
        sql = f"""
            SELECT DISTINCT
                d.teletext_id,
                d.title,
                d.content,
                d.publication_datetime
            FROM {table_name} b
            JOIN docs_teletext d
                 ON d.teletext_id = ANY(b.teletext_ids)
            WHERE b.topic = %(topic)s
              AND d.publication_datetime >= %(start)s
              AND d.publication_datetime <= %(end)s
            ORDER BY d.publication_datetime ASC
        """
        params = {"topic": top_topic, "start": start_ts, "end": end_ts}
        if limit is not None and limit > 0:
            sql += " LIMIT %(limit)s"
            params["limit"] = limit
        cur.execute(sql, params)
        rows = cur.fetchall()
    finally:
        cur.close()
        conn.close()

    results: List[Dict[str, Any]] = []
    for teletext_id, title, content, publication_datetime in rows:
        results.append(
            {
                "teletext_id": teletext_id,
                "chunk_id": None,
                "chunk_text": None,
                "title": title,
                "content": content,
                "publication_datetime": publication_datetime,
                "fts_score": None,
                "cosine_similarity": None,
                "cross_score": None,
            }
        )

    return results
