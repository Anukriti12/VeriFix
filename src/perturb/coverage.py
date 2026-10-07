"""
perturb/coverage.py - Coverage-guided curation, following Perturb & Invert's greedy selection.

P&I selects a compact subset of synthesized datapoints that maximizes joint coverage of the
tool-composition space, with the (1 - 1/e) approximation guarantee of greedy maximum coverage.
Here a datapoint "covers" each perturbation class it uses and each unordered PAIR of classes it
composes, and the greedy loop repeatedly takes the candidate that adds the most uncovered units.
"""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, List, Optional, Sequence, Set, Tuple


def units(classes: Sequence[str]) -> Set[Tuple[str, ...]]:
    cs = sorted(set(classes))
    return {(c,) for c in cs} | {tuple(p) for p in combinations(cs, 2)}


def greedy_cover(candidates: List[Sequence[str]], budget: int,
                 groups: Optional[List[str]] = None) -> List[int]:
    """Indices of up to `budget` candidates chosen greedily for coverage. If `groups` is given,
    at most one candidate per group (e.g. per source design) is chosen. Once nothing new can be
    covered, remaining slots are filled in candidate order so the subset reaches the budget."""
    covered: Set[Tuple[str, ...]] = set()
    chosen: List[int] = []
    used_groups = set()
    cand_units = [units(c) for c in candidates]
    while len(chosen) < budget:
        best, gain = None, 0
        for i, u in enumerate(cand_units):
            if i in chosen or (groups and groups[i] in used_groups):
                continue
            g = len(u - covered)
            if g > gain:
                best, gain = i, g
        if best is None:
            break
        chosen.append(best)
        covered |= cand_units[best]
        if groups:
            used_groups.add(groups[best])
    for i in range(len(candidates)):
        if len(chosen) >= budget:
            break
        if i not in chosen and not (groups and groups[i] in used_groups):
            chosen.append(i)
            if groups:
                used_groups.add(groups[i])
    return chosen


def coverage_rate(selected: Iterable[Sequence[str]], universe: Sequence[str]) -> float:
    """Fraction of all class pairs in `universe` covered by the selected datapoints."""
    allp = {tuple(p) for p in combinations(sorted(set(universe)), 2)}
    got = set()
    for s in selected:
        got |= {u for u in units(s) if len(u) == 2}
    return len(got & allp) / max(1, len(allp))
