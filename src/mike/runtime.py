"""Bounded agent loops. Invalid actions are observable failures, not repaired."""
from __future__ import annotations

import json
import time

from . import text


def run(model, question: str, tools, max_steps: int = 6) -> dict:
    """MIK-0 JSON interface."""
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


def run_text(model, question: str, tools, max_steps: int = 6, *, opening: list[dict],
             facts: list[dict] | None = None, retry: bool = False) -> dict:
    """Plain-text interface as chat turns: question, command, observation, command, ...

    Each model call, including a retry, counts against max_steps. With retry, the first invalid
    output (kept verbatim as an assistant turn) gets RETRY_MESSAGE once per task, if budget remains.
    """
    if max_steps < 1:
        raise ValueError("max_steps must be positive")
    messages = [dict(m) for m in opening] + [{"role": "user", "content": text.first_user_message(question, facts)}]
    trace = {"question": question, "steps": [], "result": None, "retries": 0}
    if facts is not None:
        trace["supplied"] = {"records": facts}
    start_wall, start_cpu = time.perf_counter(), time.process_time()
    retry_pending = False
    for index in range(max_steps):
        generation = model.generate(messages, stop=text.has_complete_command)
        step = {"generation": generation}
        if retry_pending:
            step["after_retry_message"] = True
            retry_pending = False
        trace["steps"].append(step)
        action, line = text.parse(generation["raw"])
        if action is None:
            if retry and trace["retries"] == 0 and index + 1 < max_steps:
                trace["retries"] = 1
                step["feedback"] = text.RETRY_MESSAGE
                messages += [{"role": "assistant", "content": generation["raw"]},
                             {"role": "user", "content": text.RETRY_MESSAGE}]
                retry_pending = True
                continue
            trace["result"] = {"status": "invalid_action", "value": None}
            break
        step["line"], step["action"] = line, action
        if action["action"] == "answer":
            trace["result"] = {"status": "answered", "value": action["value"], "evidence": action["evidence"]}
            break
        tool_start = time.perf_counter()
        observation = tools.execute(action)
        step["tool_seconds"] = time.perf_counter() - tool_start
        step["observation"] = observation
        step["observation_text"] = text.observation_message(action["entity"], observation, question)
        messages += [{"role": "assistant", "content": line},
                     {"role": "user", "content": step["observation_text"]}]
    if trace["result"] is None:
        trace["result"] = {"status": "budget_exhausted", "value": None}
    trace["conversation"] = messages[len(opening):]
    trace["wall_seconds"] = time.perf_counter() - start_wall
    trace["process_cpu_seconds"] = time.process_time() - start_cpu
    return trace
