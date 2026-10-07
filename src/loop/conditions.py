"""
loop/conditions.py - Condition registry. Every condition starts turn 2 from the SAME turn-1 design
and differs only in the feedback the planner receives about it (see loop/pipeline.py).
"""
CONDITIONS = {
    "B0": "no feedback: the turn-1 design is final (reference point; delta Q = 0)",
    "B1": "verbatim: the judge's critique as prose (production baseline)",
    "B2": "judge steps: critique plus fix steps written by the judge",
    "B3": "Self-Refine: the planner's own critique of its turn-1 design",
    "B4": "Reflexion: the judge's critique plus the planner's verbal reflection on it",
    "M1": "compiled: the critique turned into tool calls, unchecked",
    "M2": "verified: compiled tool calls that pass the verifier (execute, change, no new violation)",
    "M3": "retrieval: M2 plus retrieved verified fixes ranked by measured outcome",
    "M4": "retrieval_syn: M3 with synthetic multi-defect index entries",
    "ceiling": "M2 with the verifier objective replaced by the answer key (non-deployable)",
    "oracle": "the exact restore recipe as the planner's input (non-deployable)",
}
# VeriFix-D is not a separate condition: it is M3 run with the distilled planner (--model).
BASELINES = ["B0", "B1", "B2", "B3", "B4"]
METHODS = ["M1", "M2", "M3", "M4"]
CEILINGS = ["ceiling", "oracle"]
ALL = BASELINES + METHODS + CEILINGS
