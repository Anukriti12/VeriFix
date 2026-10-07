"""
perturb/fonts.py - Font style clusters for the "style" perturbation, following Perturb & Invert:
"font cluster swap via cosine co-occurrence clustering".

  1. Count, over the corpus, how often each pair of fonts appears in the same design.
  2. Represent each font by its (L2-normalized) co-occurrence row; similarity = cosine.
  3. Cluster fonts with spherical k-means (deterministic farthest-point initialization).
  4. A style perturbation moves a text element's font to the cluster whose centroid is farthest
     (cosine) from the font's own cluster, picking a frequent font there. P&I treats a cross-cluster
     swap as always significant (Table 8: "Cluster dist.: always significant").

Build the clusters from the INDEX split (or all training designs), save them, and reuse them for
evaluation. FALLBACK_CLUSTERS exist only so offline tests run without Crello.
"""
from __future__ import annotations

import json
import random
from collections import Counter
from typing import Dict, List, Optional, Tuple

import numpy as np

from ..core.design import Document

FALLBACK_CLUSTERS: Dict[str, List[str]] = {
    "sans": ["Roboto", "Open Sans", "Montserrat", "Lato", "Poppins", "Inter"],
    "serif": ["Playfair Display", "Merriweather", "Lora", "Libre Baskerville", "EB Garamond"],
    "display": ["Oswald", "Bebas Neue", "Anton", "Abril Fatface", "Alfa Slab One"],
    "script": ["Dancing Script", "Pacifico", "Great Vibes", "Satisfy", "Sacramento"],
    "handwritten": ["Caveat", "Indie Flower", "Amatic SC", "Shadows Into Light"],
    "mono": ["Roboto Mono", "Source Code Pro", "Space Mono", "IBM Plex Mono"],
}


def fonts_in(doc: Document) -> List[str]:
    return sorted({str(e.get("font")) for e in doc.elements
                   if e.get("type", "text") == "text" and e.get("font")})


class FontClusters:
    def __init__(self, fonts: List[str], labels: List[int], centroids: np.ndarray,
                 freq: Dict[str, int]):
        self.fonts = fonts
        self.labels = labels
        self.centroids = centroids
        self.freq = freq
        self._label = dict(zip(fonts, labels))

    # ---- construction ----
    @classmethod
    def from_designs(cls, designs: List[Document], k: int = 8, min_fonts: int = 12,
                     iters: int = 50) -> "FontClusters":
        freq = Counter(f for d in designs for f in fonts_in(d))
        fonts = sorted(freq)
        if len(fonts) < min_fonts:
            return cls.fallback()
        ix = {f: i for i, f in enumerate(fonts)}
        C = np.zeros((len(fonts), len(fonts)), dtype=float)
        for d in designs:
            fs = [ix[f] for f in fonts_in(d)]
            for a in fs:
                for b in fs:
                    C[a, b] += 1.0
        V = C / np.maximum(np.linalg.norm(C, axis=1, keepdims=True), 1e-12)
        k = max(2, min(k, len(fonts)))
        # deterministic farthest-point init on cosine distance, seeded by the most frequent font
        first = int(np.argmax([freq[f] for f in fonts]))
        cent = [V[first]]
        for _ in range(1, k):
            sims = np.max(np.stack([V @ c for c in cent]), axis=0)
            cent.append(V[int(np.argmin(sims))])
        cent = np.stack(cent)
        labels = np.zeros(len(fonts), dtype=int)
        for _ in range(iters):
            new = np.argmax(V @ cent.T, axis=1)
            if np.array_equal(new, labels) and _ > 0:
                break
            labels = new
            for j in range(k):
                m = V[labels == j]
                if len(m):
                    c = m.sum(axis=0)
                    cent[j] = c / max(np.linalg.norm(c), 1e-12)
        return cls(fonts, labels.tolist(), cent, dict(freq))

    @classmethod
    def fallback(cls) -> "FontClusters":
        fonts, labels = [], []
        for j, (_, fs) in enumerate(FALLBACK_CLUSTERS.items()):
            fonts += fs
            labels += [j] * len(fs)
        k = len(FALLBACK_CLUSTERS)
        return cls(fonts, labels, np.eye(k), {f: 1 for f in fonts})

    # ---- persistence ----
    def save(self, path: str) -> None:
        json.dump({"fonts": self.fonts, "labels": self.labels,
                   "centroids": self.centroids.tolist(), "freq": self.freq}, open(path, "w"))

    @classmethod
    def load(cls, path: str) -> "FontClusters":
        d = json.load(open(path))
        return cls(d["fonts"], d["labels"], np.asarray(d["centroids"]), d["freq"])

    # ---- use ----
    def cluster_of(self, font: str) -> Optional[int]:
        return self._label.get(font)

    def cluster_distance(self, a: int, b: int) -> float:
        return float(1.0 - self.centroids[a] @ self.centroids[b])

    def swap(self, font: str, rnd: random.Random) -> Tuple[Optional[str], float]:
        """A font from the farthest cluster, and the cosine distance between the two clusters."""
        src = self.cluster_of(font)
        k = self.centroids.shape[0]
        if src is None:
            src = rnd.randrange(k)
        dists = [(self.cluster_distance(src, j), j) for j in range(k) if j != src]
        dists = [x for x in dists if any(l == x[1] for l in self.labels)]
        if not dists:
            return None, 0.0
        far = max(d for d, _ in dists)
        cands = [j for d, j in dists if d >= far - 1e-9]
        tgt = rnd.choice(cands)
        pool = sorted((f for f, l in zip(self.fonts, self.labels) if l == tgt and f != font),
                      key=lambda f: -self.freq.get(f, 0))[:5]
        return (rnd.choice(pool) if pool else None), far
