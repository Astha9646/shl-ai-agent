"""
Hybrid retrieval: weighted BM25 + FAISS with title boosts and generic penalties.
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path
from typing import Any

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.retrieval.bm25_search import bm25_score_map, get_document_by_url, tokenize

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_FAISS_PATH = _PROJECT_ROOT / "data" / "faiss.index"
_METADATA_PATH = _PROJECT_ROOT / "data" / "metadata.pkl"
_MODEL_NAME = "all-MiniLM-L6-v2"

BM25_WEIGHT = 0.7
FAISS_WEIGHT = 0.3

# Extra boost when these appear in the query and match the assessment title.
PRIORITY_KEYWORDS: frozenset[str] = frozenset(
    {"java", "backend", "communication", "developer"}
)

_GENERIC_TITLE_PATTERNS: tuple[re.Pattern[str], ...] = tuple(
    re.compile(p, re.IGNORECASE)
    for p in (
        r"global\s+skills",
        r"\bai\s+skills\b",
    )
)

_QUERY_STOPWORDS: frozenset[str] = frozenset(
    {
        "the",
        "a",
        "an",
        "for",
        "with",
        "and",
        "or",
        "to",
        "in",
        "of",
        "on",
        "at",
        "by",
        "from",
        "as",
        "is",
        "are",
        "be",
        "hire",
        "hiring",
        "role",
        "senior",
        "junior",
        "level",
        "assessment",
        "assessments",
        "test",
        "tests",
        "shl",
        "need",
        "needs",
        "looking",
        "candidate",
        "candidates",
        "employee",
        "skills",
        "skill",
    }
)

TITLE_KEYWORD_BOOST = 0.22
PRIORITY_KEYWORD_BOOST = 0.30
GENERIC_PENALTY = 0.50

_index: faiss.Index | None = None
_metadata_rows: list[dict[str, Any]] | None = None
_model: SentenceTransformer | None = None


def _clean_record(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "assessment_name": row.get("assessment_name", ""),
        "assessment_url": row.get("assessment_url", ""),
        "description": row.get("description", ""),
        "test_type": row.get("test_type"),
    }


def _ensure_faiss_loaded() -> None:
    global _index, _metadata_rows, _model
    if _index is not None and _metadata_rows is not None and _model is not None:
        return
    if not _FAISS_PATH.is_file():
        raise FileNotFoundError(
            f"Missing FAISS index at {_FAISS_PATH}. Run embed.py first."
        )
    if not _METADATA_PATH.is_file():
        raise FileNotFoundError(
            f"Missing metadata at {_METADATA_PATH}. Run embed.py first."
        )
    _index = faiss.read_index(str(_FAISS_PATH))
    with _METADATA_PATH.open("rb") as f:
        payload = pickle.load(f)
    rows = payload.get("rows")
    if not isinstance(rows, list):
        raise ValueError('metadata.pkl must contain a list "rows".')
    _metadata_rows = rows
    _model = SentenceTransformer(_MODEL_NAME)


def _normalize_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = list(scores.values())
    lo, hi = min(values), max(values)
    if hi <= lo:
        return {url: (1.0 if v > 0 else 0.0) for url, v in scores.items()}
    span = hi - lo
    return {url: (v - lo) / span for url, v in scores.items()}


def _extract_query_keywords(query: str) -> list[str]:
    seen: set[str] = set()
    keywords: list[str] = []
    for tok in tokenize(query):
        if tok in _QUERY_STOPWORDS or len(tok) < 2:
            continue
        if tok not in seen:
            seen.add(tok)
            keywords.append(tok)
    return keywords


def _title_keyword_boost(title: str, keywords: list[str]) -> float:
    title_lower = title.lower()
    boost = 0.0
    for kw in keywords:
        if not re.search(rf"\b{re.escape(kw)}\b", title_lower):
            continue
        boost += TITLE_KEYWORD_BOOST
        if kw in PRIORITY_KEYWORDS:
            boost += PRIORITY_KEYWORD_BOOST
    return boost


def _generic_penalty(title: str) -> float:
    for pattern in _GENERIC_TITLE_PATTERNS:
        if pattern.search(title):
            return GENERIC_PENALTY
    return 0.0


def _faiss_score_map(query: str, top_k: int) -> dict[str, tuple[float, dict[str, Any]]]:
    _ensure_faiss_loaded()
    assert _index is not None and _metadata_rows is not None and _model is not None

    q = (query or "").strip()
    if not q:
        return {}

    q_emb = _model.encode(
        q,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).astype(np.float32, copy=False)
    q_emb = q_emb.reshape(1, -1)

    n = len(_metadata_rows)
    k = max(0, min(top_k, n))
    if k == 0:
        return {}

    sims, idxs = _index.search(q_emb, k)
    out: dict[str, tuple[float, dict[str, Any]]] = {}
    for j in range(k):
        i = int(idxs[0, j])
        if i < 0 or i >= n:
            continue
        doc = _clean_record(_metadata_rows[i])
        url = str(doc["assessment_url"])
        out[url] = (float(sims[0, j]), doc)
    return out


def _top_urls(scores: dict[str, float], limit: int) -> set[str]:
    if limit <= 0 or not scores:
        return set()
    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return {url for url, _ in ranked[:limit]}


def _resolve_document(
    url: str,
    faiss_map: dict[str, tuple[float, dict[str, Any]]],
) -> dict[str, Any]:
    if url in faiss_map:
        return faiss_map[url][1]
    doc = get_document_by_url(url)
    if doc:
        return _clean_record(doc)
    return {
        "assessment_name": "",
        "assessment_url": url,
        "description": "",
        "test_type": None,
    }


def hybrid_search(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Rank assessments with weighted BM25 (0.7) + FAISS (0.3), title keyword boosts,
    and penalties for generic catalog titles (e.g. Global Skills, AI Skills).
    """
    q = (query or "").strip()
    if not q:
        return []

    pool = max(top_k * 8, 24)
    keywords = _extract_query_keywords(q)

    bm25_raw = bm25_score_map(q)
    if not bm25_raw:
        return []

    faiss_map = _faiss_score_map(q, pool)
    candidates = _top_urls(bm25_raw, pool) | set(faiss_map.keys())

    bm25_norm = _normalize_scores({url: bm25_raw.get(url, 0.0) for url in candidates})
    faiss_norm = _normalize_scores(
        {url: faiss_map[url][0] if url in faiss_map else 0.0 for url in candidates}
    )

    ranked: list[tuple[float, dict[str, Any]]] = []
    for url in candidates:
        doc = _resolve_document(url, faiss_map)
        title = str(doc.get("assessment_name", ""))

        base = (
            BM25_WEIGHT * bm25_norm.get(url, 0.0)
            + FAISS_WEIGHT * faiss_norm.get(url, 0.0)
        )
        title_boost = _title_keyword_boost(title, keywords)
        penalty = _generic_penalty(title)
        final = base + title_boost - penalty

        ranked.append((final, doc))

    ranked.sort(
        key=lambda item: (
            item[0],
            _title_keyword_boost(str(item[1].get("assessment_name", "")), keywords),
            bm25_norm.get(str(item[1].get("assessment_url", "")), 0.0),
        ),
        reverse=True,
    )

    out: list[dict[str, Any]] = []
    for _score, doc in ranked[:top_k]:
        out.append(_clean_record(doc))
    return out
