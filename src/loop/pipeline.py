"""
loop/pipeline.py - Turn 1 -> turn 2 protocol for every condition (plus a multi-pass variant).

PRIMARY PROTOCOL (two_turn):
  turn 1 : the planner sees the degraded design Y and the request, and edits once -> D1.
           Turn 1 is computed ONCE per case and SHARED by every condition, so all conditions
           start turn 2 from the identical design. The comparison is paired.
  turn 2 : the condition decides what feedback the planner gets about D1 (raw critique, judge
           steps, self-critique, compiled edits, checked edits, retrieved fixes, ...); the planner
           edits once more -> D2.
  primary metric : Q(D2), and the per-case gain dQ = Q(D2) - Q(D1).
Why fixed turns instead of "loop until the judge is satisfied": the judge's stopping decision is
noisy and would change how many passes each condition gets; one feedback turn gives every
condition the same budget and isolates the quality of the feedback itself.

MULTI-PASS (multi_turn): repeat the feedback step up to max_passes, stopping when the judge's mean
score reaches the threshold. Used only for the quality-versus-budget analysis.

BOUNDARY: the agent sees only renders and the request; the verifier sees the current design, the
tool set, and Phi. X and the targets are used only to score after the episode, except in the
'ceiling' and 'oracle' conditions, which are labeled non-deployable.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from ..agents.judge import JUDGE_DIMS, critique_text
from ..agents.verifier import (format_for_planner, log_summary, verify, verify_against_truth)
from ..core.design import Document
from ..core.metrics import collateral, exact_rate, perceptual_distance, quality
from ..core.objective import design_penalty
from ..core.tools import execute_actions
from ..perturb.operators import restore_recipe
from ..retrieval import dense, outcome, sparse
from ..retrieval.index import FixIndex, detect_defects

COMPILE = {"M1", "M2", "M3", "M4", "ceiling"}
VERIFY = {"M2", "M3", "M4"}
RETRIEVE = {"M3", "M4"}

_LEARNED_SINGLETON = None


def _default_learned():
    global _LEARNED_SINGLETON
    if _LEARNED_SINGLETON is None:
        from ..retrieval.learned import LearnedRanker
        _LEARNED_SINGLETON = LearnedRanker.load(None)
    return _LEARNED_SINGLETON


class Pipeline:
    def __init__(self, planner, judge, compiler=None, index: Optional[FixIndex] = None,
                 max_passes: int = 3, judge_threshold: float = 1.0, retriever: str = "learned",
                 lam: float = 0.5, embed_client=None, key_mode: str = "both",
                 learned_ranker=None, top_k: int = 3):
        self.planner, self.judge, self.compiler, self.index = planner, judge, compiler, index
        self.max_passes, self.judge_threshold = max_passes, judge_threshold
        self.retriever, self.lam, self.embed_client = retriever, lam, embed_client
        self.key_mode, self.learned_ranker, self.top_k = key_mode, learned_ranker, top_k

    # ------------------------------------------------------------ retrieval (M3/M4)
    def retrieve(self, jr: Dict):
        if not self.index:
            return "", []
        defects = detect_defects(jr)
        cands = self.index.candidates(defects) if defects else []
        if not cands:
            return "", []
        crit, km, k = critique_text(jr), self.key_mode, self.top_k
        if self.retriever == "sparse":
            ranked = sparse.rank(crit, cands, top_k=k, key_mode=km)
        elif self.retriever == "dense":
            ranked = dense.rank(crit, cands, top_k=k, client=self.embed_client, key_mode=km)
        elif self.retriever == "outcome":
            ranked = outcome.rank(crit, cands, top_k=k, lam=self.lam, key_mode=km)
        else:
            ranked = (self.learned_ranker or _default_learned()).rank(crit, cands, top_k=k, key_mode=km)
        if not ranked:
            return "", []
        lines = ["## KNOWN GOOD FIXES (from similar past cases; adapt the values to this design)"]
        for _, ex in ranked:
            lines.append(f"- for: {ex.critique} (measured gain {ex.utility:+.2f})")
            for a in ex.actions[:3]:
                params = ", ".join(f"{p}={v}" for p, v in (a.get("params") or {}).items())
                lines.append(f"    {a['action']}: {params}")
        return "\n".join(lines), [(crit, ex) for _, ex in ranked]

    # ------------------------------------------------------------ one feedback step
    def feedback_step(self, D: Document, jr: Dict, query: str, condition: str,
                      targets: Optional[List[Dict]] = None, X: Optional[Document] = None,
                      memory: str = "") -> Dict:
        """Build the condition's feedback about design D (judged as jr), let the planner edit
        once, and return the new design plus what was shown and checked."""
        fb, extra, vlog, injected = None, None, [], []
        if condition == "B1":
            fb = jr["feedback_text"]
        elif condition == "B2":
            fb = jr["feedback_text"] + "\n\nSuggested steps:\n" + judge_steps(self.judge, D, jr)
        elif condition == "B3":
            fb = "Your own critique of the current design:\n" + self_critique(self.planner, D)
        elif condition == "B4":
            fb = jr["feedback_text"]
            extra = "## YOUR REFLECTION\n" + (memory or reflect(self.planner, D, jr["feedback_text"]))
        elif condition in COMPILE:
            proposals = self.compiler.compile(D, jr) if self.compiler else []
            if condition == "ceiling":
                kept, vlog = verify_against_truth(D, proposals, X, targets or [])
            elif condition in VERIFY:
                kept, vlog = verify(D, proposals)
            else:  # M1: unchecked
                kept = [{"action": p.get("action"), "target": p.get("target"),
                         "params": p.get("params", {})} for p in proposals]
            if condition == "M1":
                extra = format_for_planner(kept, D, header="SUGGESTED EDITS (from the critique, unchecked)")
            else:
                extra = format_for_planner(kept, D)
            if condition in RETRIEVE:
                blk, injected = self.retrieve(jr)
                if blk:
                    extra += "\n\n" + blk
        elif condition == "oracle":
            extra = format_for_planner(restore_recipe(D, targets or []), D,
                                       header="EXACT FIX (apply these edits)")
        if condition == "B0":
            return {"doc": D, "actions": [], "verifier": {}, "injected": []}
        actions = self.planner.plan(D, query, feedback=fb, extra_block=extra)
        return {"doc": execute_actions(D, actions), "actions": actions,
                "verifier": log_summary(vlog) if vlog else {}, "injected": injected}

    # ------------------------------------------------------------ protocol
    def run_turn1(self, Y: Document, query: str) -> Dict:
        actions = self.planner.plan(Y, query)
        D1 = execute_actions(Y, actions)
        return {"doc": D1, "actions": actions, "judge": self.judge.score(D1, query)}

    def run(self, Y: Document, query: str, condition: str, targets: Optional[List[Dict]] = None,
            X: Optional[Document] = None, turn1: Optional[Dict] = None) -> Dict:
        """Two-turn protocol. Pass the shared turn1 (from run_turn1) for paired comparisons."""
        t1 = turn1 or self.run_turn1(Y, query)
        D1, jr1 = t1["doc"], t1["judge"]
        step = self.feedback_step(D1, jr1, query, condition, targets, X)
        D2 = step["doc"]
        jr2 = jr1 if condition == "B0" else self.judge.score(D2, query)
        return self._result(condition, Y, D1, D2, jr1, jr2, targets, X, step, n_turns=2)

    def run_multi(self, Y: Document, query: str, condition: str,
                  targets: Optional[List[Dict]] = None, X: Optional[Document] = None,
                  turn1: Optional[Dict] = None) -> Dict:
        """Multi-pass variant for the budget analysis: feedback steps until the judge's mean
        score reaches the threshold or max_passes turns have been taken."""
        t1 = turn1 or self.run_turn1(Y, query)
        D, jr = t1["doc"], t1["judge"]
        history = [{"turn": 1, "quality": quality(D, targets) if targets is not None else None,
                    "judge": jr["overall"]}]
        step, mem = {"verifier": {}, "injected": [], "actions": []}, ""
        for turn in range(2, self.max_passes + 1):
            if jr["overall"] >= self.judge_threshold or condition == "B0":
                break
            step = self.feedback_step(D, jr, query, condition, targets, X, memory=mem)
            D = step["doc"]
            jr = self.judge.score(D, query)
            if condition == "B4":
                mem += f"\nTurn {turn}: {jr['feedback_text'][:200]}"
            history.append({"turn": turn, "quality": quality(D, targets) if targets is not None else None,
                            "judge": jr["overall"]})
        out = self._result(condition, Y, t1["doc"], D, t1["judge"], jr, targets, X, step,
                           n_turns=len(history))
        out["history"] = history
        return out

    def _result(self, condition, Y, D1, D2, jr1, jr2, targets, X, step, n_turns) -> Dict:
        out = {"condition": condition, "n_turns": n_turns,
               "judge_turn1": jr1["overall"], "judge_turn2": jr2["overall"],
               "judge_scores_turn2": {d: jr2[d]["score"] for d in JUDGE_DIMS},
               "phi_turn1": round(design_penalty(D1)[0], 4),
               "phi_turn2": round(design_penalty(D2)[0], 4),
               "verifier": step.get("verifier", {}), "n_edits_turn2": len(step.get("actions", [])),
               "final_doc": D2}
        if targets is not None:
            q0, q1, q2 = quality(Y, targets), quality(D1, targets), quality(D2, targets)
            out.update({"q_input": q0, "q_turn1": q1, "q_turn2": q2, "delta_q": q2 - q1,
                        "exact_turn2": exact_rate(D2, targets)})
        if X is not None:
            out["collateral"] = collateral(D2, X, targets or [])
            out["perceptual_distance"] = perceptual_distance(D2, X)
        if step.get("injected") and targets is not None:
            out["_injected"] = step["injected"]
            out["_realized_delta"] = out["delta_q"]
        return out


# ---------------------------------------------------------------- helper model calls
def judge_steps(judge, doc, jr) -> str:
    from ..agents.llm_client import chat
    from ..core.render import render
    sys = ("You are the same design judge. Given your critique, write a short numbered list of "
           "imperative fix steps a designer should take. Plain text, no JSON.")
    crit = "\n".join(f"{d}: {jr.get(d, {}).get('explanation', '')}" for d in JUDGE_DIMS)
    return chat(judge.client, judge.model, sys, f"Critique:\n{crit}",
                images=[render(doc, marked=False, fonts_dir=judge.fonts_dir)],
                counter=judge.counter, temperature=0.2, max_tokens=400)


def self_critique(planner, doc) -> str:
    from ..agents.llm_client import chat
    from ..core.render import render
    sys = ("Critique this design in 2-3 sentences: what is wrong with its layout, typography, "
           "and color? Plain text.")
    return chat(planner.client, planner.model, sys, "Critique the shown design.",
                images=[render(doc, marked=False, fonts_dir=planner.fonts_dir)],
                counter=planner.counter, temperature=0.3, max_tokens=200)


def reflect(planner, doc, feedback: str) -> str:
    from ..agents.llm_client import chat
    sys = ("You edited a design and received the feedback below. In 2-3 sentences, reflect on what "
           "your previous edits got wrong and what you will do differently. Plain text.")
    return chat(planner.client, planner.model, sys, feedback, images=None,
                counter=planner.counter, temperature=0.3, max_tokens=200)
