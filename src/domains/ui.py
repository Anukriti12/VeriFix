"""
domains/ui.py - STUB. Mobile/UI layout (RICO).

Implement load() to return core.Document objects from RICO view hierarchies: each view node
-> an element with left/top/width/height and a type (text/button/image). Accessibility defects
(contrast, touch-target size) map cleanly onto the X-free objective and the HCI angle.
Compare against UI critique / layout baselines.
"""
from __future__ import annotations
from typing import List
from ..core.design import Document
from .base import Domain

class UIDomain(Domain):
    name = "ui"
    def load(self, n: int, split: str = "test") -> List[Document]:
        raise NotImplementedError("Implement RICO loader (see docstring).")
