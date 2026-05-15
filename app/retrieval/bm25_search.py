"""
BM25 retrieval over SHL assessments (assessment_name + description).
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from rank_bm25 import BM25Okapi

_PROJECT_ROOT = Path(__file__).resolve().parents[2]
_ASSESSMENTS_PATH = _PROJECT_ROOT / "data" / "assessments.json"

_bm25: BM25Okapi | None = None
_documents: list[dict[str, Any]] | None = None
_url_to_index: dict[str, int] | None = None


def tokenize(text: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", text.lower())


def _build_corpus() -> tuple[list[list[str]], list[dict[str, Any]]]:
    with _ASSESSMENTS_PATH.open(encoding="utf-8") as f:
        data = json.load(f)
    assessments = data.get("assessments")
    if not isinstance(assessments, list):
        raise ValueError('Expected JSON key "assessments" to be a list.')

    tokenized: list[list[str]] = []
    docs: list[dict[str, Any]] = []
    for row in assessments:
        name = (row.get("assessment_name") or "").strip()
        desc = (row.get("description") or "").strip()
        combined = f"{name} {desc}".strip()
        tokenized.append(tokenize(combined))
        docs.append(
            {
                "assessment_name": row.get("assessment_name", ""),
                "assessment_url": row.get("assessment_url", ""),
                "description": row.get("description", ""),
                "test_type": row.get("test_type"),
            }
        )
    return tokenized, docs


def _ensure_loaded() -> None:
    global _bm25, _documents, _url_to_index
    if _bm25 is not None and _documents is not None:
        return
    tokenized, docs = _build_corpus()
    _documents = docs
    _bm25 = BM25Okapi(tokenized)
    _url_to_index = {
        str(doc["assessment_url"]): i for i, doc in enumerate(_documents)
    }


def bm25_score_map(query: str) -> dict[str, float]:
    """Return BM25 scores keyed by ``assessment_url``."""
    _ensure_loaded()
    assert _bm25 is not None and _documents is not None

    q_tokens = tokenize(query)
    if not q_tokens:
        return {}

    raw = _bm25.get_scores(q_tokens)
    return {
        str(_documents[i]["assessment_url"]): float(raw[i])
        for i in range(len(_documents))
    }


def get_document_by_url(url: str) -> dict[str, Any] | None:
    _ensure_loaded()
    assert _documents is not None and _url_to_index is not None
    idx = _url_to_index.get(url)
    if idx is None:
        return None
    return {**_documents[idx]}


def search_bm25(query: str, top_k: int = 5) -> list[dict[str, Any]]:
    """
    Return the top ``top_k`` assessments by BM25 score for ``query``.

    Scoring uses the concatenation of assessment_name and description.
    """
    _ensure_loaded()
    assert _bm25 is not None and _documents is not None

    q_tokens = tokenize(query)
    if not q_tokens:
        return []

    scores = _bm25.get_scores(q_tokens)
    n = len(scores)
    k = max(0, min(top_k, n))
    ranked = sorted(range(n), key=lambda i: scores[i], reverse=True)[:k]
    return [{**_documents[i]} for i in ranked]
