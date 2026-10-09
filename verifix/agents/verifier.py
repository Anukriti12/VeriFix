"""
agents/verifier.py - The verifier (M2). Pure code, tested offline.

Plain-language description (use this wording in the paper):
  For each edit the compiler proposes, the verifier applies it to a COPY of the current design,
  then keeps it only if (a) the edit executes (the target exists, the tool applies to it, the
  parameters are valid), (b) it changes the design, and (c) the reference-free objective Phi does
  not get worse. Accepted edits accumulate on the copy, so later edits are checked against the
  result of earlier ones. The real design is never touched; the planner receives only the
  accepted edits, and keeps the final say over what is applied.

Phi checks absolute standards (contrast, size floor, tilt, bounds, overlap, occlusion), so the
verifier is a HARM FILTER: it removes compiler edits that create new problems. It cannot confirm
that an edit restored the original design, because Phi cannot see most intent-relative defects.

verify_against_truth() is the CEILING variant: same rule, but the objective uses the answer key
(unclipped distance to X plus collateral). Not deployable; it bounds how good a verifier objective could be.
"""
from __future__ import annotations

from typing import Callable, Dict, List, Optional, Tuple

from ..core.design import Document, structural_diff
from ..core.metrics import collateral, truth_distance
from ..core.objective import design_penalty
from ..core.tools import execute_tool

Objective = Callable[[Document], float]


def phi(doc: Document) -> float:
    return design_penalty(doc)[0]


def verify(doc: Document, proposals: List[Dict], objective: Optional[Objective] = phi,
           eps: float = 1e-9) -> Tuple[List[Dict], List[Dict]]:
    """Return (accepted calls, per-proposal log). objective=None checks feasibility only."""
    cur = doc
    kept: List[Dict] = []
    log: List[Dict] = []
    for p in proposals:
        call = {"action": p.get("action"), "target": p.get("target"), "params": p.get("params", {})}
        rec = {"action": call["action"], "target": call["target"], "accepted": False}
        after, why = execute_tool(cur, call)
        if after is None:
            rec["reason"] = why; log.append(rec); continue
        if not structural_diff(after, cur):
            rec["reason"] = "no_change"; log.append(rec); continue
        if objective is not None:
            before_v, after_v = objective(cur), objective(after)
            rec["phi_before"], rec["phi_after"] = round(before_v, 4), round(after_v, 4)
            if after_v > before_v + eps:
                rec["reason"] = "worsens_objective"; log.append(rec); continue
        kept.append(call)
        rec["accepted"], rec["reason"] = True, "accepted"
        log.append(rec)
        cur = after
    return kept, log


def truth_objective(X: Document, targets: List[Dict]) -> Objective:
    """CEILING ONLY: unclipped distance to the answer key plus collateral (per target)."""
    n = max(1, len(targets))
    return lambda d: truth_distance(d, targets) + collateral(d, X, targets) / n


def verify_against_truth(doc: Document, proposals: List[Dict], X: Document,
                         targets: List[Dict]) -> Tuple[List[Dict], List[Dict]]:
    return verify(doc, proposals, objective=truth_objective(X, targets))


def log_summary(log: List[Dict]) -> Dict:
    out = {"proposed": len(log), "accepted": sum(1 for r in log if r["accepted"])}
    for r in log:
        if not r["accepted"]:
            out[r["reason"]] = out.get(r["reason"], 0) + 1
    return out


def format_for_planner(calls: List[Dict], doc: Document,
                       header: str = "CHECKED EDITS (they execute and create no new design problems)"
                       ) -> str:
    inv = {e["id"]: str(i) for i, e in enumerate(doc.elements)}
    lines = [f"## {header}"]
    if not calls:
        lines.append("- (none)")
    for c in calls:
        num = inv.get(c["target"], c["target"])
        params = ", ".join(f"{k}={v}" for k, v in (c.get("params") or {}).items())
        lines.append(f"- {c['action']} on {num}: {params}")
    return "\n".join(lines)
