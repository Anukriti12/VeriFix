"""
retrieval/dense.py - Dense bi-encoder ranking (Approach A2). OPTIONAL, UNTESTED here.

Embeds exemplar critiques and the live critique, ranks by cosine. The point is paraphrase
robustness ("muddled hierarchy" ~ "competing focal points"), which lexical BM25 misses.

Backends, tried in order:
  1. sentence-transformers (local, no API)        -> set VERIFIX_ST_MODEL (default MiniLM)
  2. an OpenAI-compatible embeddings endpoint      -> set VERIFIX_EMBED_MODEL + the client

IMPORTANT (lesson from the earlier internal A2 run): if no backend is available this RAISES
rather than silently falling back to lexical, so you never report a "dense" number that was
actually BM25. Confirm the backend engaged before trusting any dense result.
"""
from __future__ import annotations

import os
from typing import List, Optional, Tuple

import numpy as np

from .index import Exemplar

_st_model = None


def _embed_st(texts: List[str]) -> np.ndarray:
    global _st_model
    from sentence_transformers import SentenceTransformer  # type: ignore
    if _st_model is None:
        name = os.environ.get("VERIFIX_ST_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
        _st_model = SentenceTransformer(name)
    return np.asarray(_st_model.encode(texts, normalize_embeddings=True), dtype=np.float32)


def _embed_api(texts: List[str], client, model: str) -> np.ndarray:
    resp = client.embeddings.create(model=model, input=texts)
    vecs = np.asarray([d.embedding for d in resp.data], dtype=np.float32)
    vecs /= (np.linalg.norm(vecs, axis=1, keepdims=True) + 1e-9)
    return vecs


def _embed(texts: List[str], client=None) -> np.ndarray:
    try:
        return _embed_st(texts)
    except Exception:
        pass
    model = os.environ.get("VERIFIX_EMBED_MODEL")
    if client is not None and model:
        return _embed_api(texts, client, model)
    raise RuntimeError(
        "dense retrieval has no backend: install sentence-transformers, or set "
        "VERIFIX_EMBED_MODEL and pass an embeddings client. Refusing to fall back to lexical."
    )


def rank(query_critique: str, candidates: List[Exemplar], top_k: int = 3, client=None,
         key_mode: str = "critique") -> List[Tuple[float, Exemplar]]:
    from .index import exemplar_key_text
    if not candidates:
        return []
    mat = _embed([query_critique] + [exemplar_key_text(e, key_mode) for e in candidates],
                 client=client)
    q, M = mat[0], mat[1:]
    sims = (M @ q).tolist()
    ranked = sorted(zip(sims, candidates), key=lambda x: -x[0])
    return [(float(s), e) for s, e in ranked[:top_k]]
