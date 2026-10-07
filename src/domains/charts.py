"""
domains/charts.py - STUB. Charts with EXACT executable ground truth (ChartMimic-style).

This domain is attractive because the inverse is exact: perturb a chart spec (axis range,
label, series value, color) and the fix is the exact inverse edit, verified by re-rendering
the chart code. Represent the chart spec as a Document (elements = spec fields) OR keep the
code and wrap render() to execute it. Connect to MM-ReCoder (CVPR'26) / METAL (ACL'25).
Executable verification here is a strong CVPR point (reference-free check is a real renderer).
"""
from __future__ import annotations
from typing import List
from ..core.design import Document
from .base import Domain

class ChartDomain(Domain):
    name = "charts"
    def load(self, n: int, split: str = "test") -> List[Document]:
        raise NotImplementedError("Implement chart-spec loader + code renderer (see docstring).")
