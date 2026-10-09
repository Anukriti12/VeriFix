"""
domains/base.py - The Domain interface. Add a visual domain by implementing this.

The CVPR story needs >= 3 domains (graphic design + posters/UI/slides/charts). Each domain
supplies clean documents in the SAME core.Document representation, so the perturb-and-invert
generator, the executor, the X-free objective, the verifier, and the retrieval index are all
domain-agnostic and reused unchanged. A new domain only has to:
  - load() clean Documents from its dataset
  - (optionally) override the perturbation class set and the objective terms that apply
  - ensure its elements use the shared property schema (left/top/width/height/font/... )

See graphic_design.py for a full implementation and posters/ui/charts for stub contracts.
"""
from __future__ import annotations

from typing import List, Optional

from ..core.design import Document
from ..perturb.operators import ALL_CLASSES


class Domain:
    name: str = "base"
    # which perturbation classes are meaningful in this domain (default: all)
    perturb_classes: List[str] = list(ALL_CLASSES)

    def load(self, n: int, split: str = "test") -> List[Document]:
        raise NotImplementedError

    def real_defects(self, n: int, split: str = "test") -> List[Document]:
        """OPTIONAL but important for CVPR: return documents with REAL (not synthetic)
        defects -- e.g. outputs of a layout generator, or human-flagged issues -- for the
        held-out transfer evaluation that answers 'does this only undo your own perturbations?'
        Return [] if not available yet."""
        return []
