"""Bounded agent loop. Invalid actions are observable failures, not repaired."""
from __future__ import annotations

import json
import time


def run(model, question: str, tools, max_steps: int = 6) -> dict:
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    state = {"question": question, "history": []}
    trace = {"question": question, "steps": [], "result": None}
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    for _ in range(max_steps):
        generation = model.decide(state)
        step = {"generation": generation}
        trace["steps"].append(step)
        try:
            action = json.loads(generation["raw"])
            if not isinstance(action, dict) or not isinstance(action.get("action"), str):
                raise ValueError("expected an action object")
        except (json.JSONDecodeError, ValueError):
            trace["result"] = {"status": "invalid_action", "value": None}
            break
        step["action"] = action
        if action["action"] == "answer":
            evidence = action.get("evidence", [])
            if not (action.get("value") is None or isinstance(action.get("value"), str)) or not isinstance(evidence, list) or not all(isinstance(v, str) for v in evidence):
                trace["result"] = {"status": "invalid_action", "value": None}
            else:
                trace["result"] = {"status": "answered", "value": action.get("value"), "evidence": evidence}
            break
        tool_start = time.perf_counter()
        observation = tools.execute(action)
        step["tool_seconds"] = time.perf_counter() - tool_start
        step["observation"] = observation
        state["history"].append({"action": action, "observation": observation})
    if trace["result"] is None:
        trace["result"] = {"status": "budget_exhausted", "value": None}
    trace["wall_seconds"] = time.perf_counter() - start_wall
    trace["process_cpu_seconds"] = time.process_time() - start_cpu
    return trace
