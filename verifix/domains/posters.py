"""
domains/posters.py - STUB. Poster layout (PKU PosterLayout / CGL).

Implement load() to return core.Document objects from PKU PosterLayout or CGL: background
image as an 'image' element (asset), text/logo boxes as elements with the shared schema.
Compare against RALF (CVPR'24), CAL-RAG, PosterReward (CVPR'26) on standard layout metrics
(underlay, alignment, overlap). perturb_classes can stay the default set.
"""
from __future__ import annotations
from typing import List
from ..core.design import Document
from .base import Domain

class PosterDomain(Domain):
    name = "posters"
    def load(self, n: int, split: str = "test") -> List[Document]:
        raise NotImplementedError("Implement PKU PosterLayout/CGL loader (see docstring).")
