"""Evaluation metrics and task-specific transparent symbolic baseline."""
from __future__ import annotations

import math
from collections import Counter

from .runtime import run
from .tools import Tools


def evaluate_one(model, task, records, enabled, max_steps):
    trace = run(model, task["question"], Tools(records, enabled), max_steps)
    trace["task_id"] = task["id"]
    trace["tools_enabled"] = enabled
    result = trace["result"]
    observed_ids = {r["id"] for step in trace["steps"]
                    for r in step.get("observation", {}).get("records", [])}
    cited = set(result.get("evidence", []))
    trace["score"] = {
        "correct": result.get("status") == "answered" and result.get("value") == task["answer"],
        "complete_evidence": set(task["evidence"]).issubset(cited) and cited.issubset(observed_ids),
        "abstained": result.get("status") == "answered" and result.get("value") is None,
    }
    return trace


def wilson(successes, n):
    if not n:
        return [None, None]
    z = 1.959963984540054
    p = successes / n
    denominator = 1 + z*z/n
    center = (p + z*z/(2*n))/denominator
    half = z*math.sqrt(p*(1-p)/n + z*z/(4*n*n))/denominator
    return [max(0., center-half), min(1., center+half)]


def summarize(traces):
    n = len(traces)
    correct = sum(t["score"]["correct"] for t in traces)
    return {"tasks": n, "correct": correct, "accuracy": correct/n if n else None,
            "accuracy_wilson_95": wilson(correct, n),
            "correct_with_complete_evidence": sum(t["score"]["correct"] and t["score"]["complete_evidence"] for t in traces),
            "abstentions": sum(t["score"]["abstained"] for t in traces),
            "statuses": dict(Counter(t["result"]["status"] for t in traces)),
            "searches": sum(s.get("action", {}).get("action") == "search" for t in traces for s in t["steps"]),
            "input_tokens": sum(s["generation"]["input_tokens"] for t in traces for s in t["steps"]),
            "output_tokens": sum(s["generation"]["output_tokens"] for t in traces for s in t["steps"]),
            "wall_seconds": sum(t["wall_seconds"] for t in traces),
            "process_cpu_seconds": sum(t["process_cpu_seconds"] for t in traces)}


def symbolic_baseline(task, records, enabled):
    """Task-specific four-relation traversal; deliberately not a general agent."""
    tools = Tools(records, enabled)
    entity, evidence = task["start"], []
    for relation in ("inside", "belongs_to", "located_in", "has_property"):
        matches = [r for r in tools.execute({"action": "search", "entity": entity}).get("records", [])
                   if r["relation"] == relation]
        if len(matches) != 1:
            return {"value": None, "evidence": evidence}
        evidence.append(matches[0]["id"])
        entity = matches[0]["object"]
    return {"value": entity, "evidence": evidence}


def compare_runs(runs):
    """Compare only paired task IDs; explicitly retain incomplete runs."""
    result = {"runs": [], "paired": []}
    for label, metadata, traces, complete in runs:
        result["runs"].append({"label": label, "model": metadata["model"],
                               "world_sha256": metadata["world_sha256"],
                               "tools_enabled": metadata["tools_enabled"],
                               "complete": complete, "summary": summarize(traces)})
    for i, (left_label, left_metadata, left_traces, _) in enumerate(runs):
        for right_label, right_metadata, right_traces, _ in runs[i + 1:]:
            if left_metadata["world_sha256"] != right_metadata["world_sha256"]:
                continue
            left = {t["task_id"]: t for t in left_traces}
            right = {t["task_id"]: t for t in right_traces}
            shared = sorted(left.keys() & right.keys())
            result["paired"].append({"left": left_label, "right": right_label,
                "tasks": len(shared),
                "both_correct": sum(left[k]["score"]["correct"] and right[k]["score"]["correct"] for k in shared),
                "only_left_correct": sum(left[k]["score"]["correct"] and not right[k]["score"]["correct"] for k in shared),
                "only_right_correct": sum(right[k]["score"]["correct"] and not left[k]["score"]["correct"] for k in shared)})
    return result
