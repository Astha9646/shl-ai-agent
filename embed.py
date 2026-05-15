"""
Build FAISS index from SHL assessments using sentence-transformers.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

ROOT = Path(__file__).resolve().parent
ASSESSMENTS_PATH = ROOT / "data" / "assessments.json"
FAISS_PATH = ROOT / "data" / "faiss.index"
METADATA_PATH = ROOT / "data" / "metadata.pkl"

MODEL_NAME = "all-MiniLM-L6-v2"
ENCODE_BATCH_SIZE = 64


def load_assessments(path: Path) -> list[dict]:
    print(f"Loading assessments from {path} …", flush=True)
    with path.open(encoding="utf-8") as f:
        data = json.load(f)
    assessments = data.get("assessments")
    if not isinstance(assessments, list):
        raise ValueError('Expected top-level key "assessments" to be a list.')
    print(f"Found {len(assessments)} assessments (count field: {data.get('count')!r}).", flush=True)
    return assessments


def build_texts(assessments: list[dict]) -> tuple[list[str], list[dict]]:
    texts: list[str] = []
    metadata_rows: list[dict] = []
    for i, row in enumerate(assessments):
        name = (row.get("assessment_name") or "").strip()
        desc = (row.get("description") or "").strip()
        text = f"{name}\n{desc}".strip() if desc else name
        texts.append(text)
        metadata_rows.append(
            {
                "index": i,
                "assessment_name": row.get("assessment_name", ""),
                "assessment_url": row.get("assessment_url", ""),
                "description": row.get("description", ""),
                "test_type": row.get("test_type"),
            }
        )
    return texts, metadata_rows


def encode_in_batches(
    model: SentenceTransformer, texts: list[str], batch_size: int
) -> np.ndarray:
    n = len(texts)
    dim = (
        model.get_embedding_dimension()
        if hasattr(model, "get_embedding_dimension")
        else model.get_sentence_embedding_dimension()
    )
    out = np.zeros((n, dim), dtype=np.float32)
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = texts[start:end]
        print(f"Encoding batch rows [{start}:{end}) / {n} …", flush=True)
        emb = model.encode(
            batch,
            batch_size=batch_size,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        out[start:end] = emb.astype(np.float32, copy=False)
    print(f"Encoded {n} vectors of dimension {dim}.", flush=True)
    return out


def main() -> None:
    assessments = load_assessments(ASSESSMENTS_PATH)
    texts, metadata_rows = build_texts(assessments)

    print(f"Loading model {MODEL_NAME!r} …", flush=True)
    model = SentenceTransformer(MODEL_NAME)

    embeddings = encode_in_batches(model, texts, ENCODE_BATCH_SIZE)

    dim = embeddings.shape[1]
    print(f"Building FAISS IndexFlatIP (dim={dim}) …", flush=True)
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)

    FAISS_PATH.parent.mkdir(parents=True, exist_ok=True)
    faiss.write_index(index, str(FAISS_PATH))
    print(f"Saved FAISS index to {FAISS_PATH}", flush=True)

    meta_payload = {
        "model_name": MODEL_NAME,
        "faiss_index_path": str(FAISS_PATH.name),
        "embedding_text_fields": ["assessment_name", "description"],
        "num_vectors": int(embeddings.shape[0]),
        "dimension": dim,
        "rows": metadata_rows,
    }
    with METADATA_PATH.open("wb") as f:
        pickle.dump(meta_payload, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved metadata pickle to {METADATA_PATH} ({len(metadata_rows)} rows).", flush=True)


if __name__ == "__main__":
    main()
