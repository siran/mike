"""Failure analysis of evaluation traces against the world's answer key (read-only)."""
from __future__ import annotations

import re
from collections import Counter

HOPS = ("object", "box", "station", "region", "property")
_IDENTIFIER = re.compile(r"\b[A-Z][0-9A-F]{12}(?:[0-9A-F]{4})?\b")


def opening_identifiers(opening: list[dict] | None) -> set[str]:
    """Identifiers shown in a run's opening (its worked example), from the run's own metadata."""
    if not opening:
        return set()
    return {m for message in opening for m in _IDENTIFIER.findall(message["content"])}


def chain_of(task: dict, records: list[dict]) -> list[dict]:
    by_id = {r["id"]: r for r in records}
    return [by_id[i] for i in task["evidence"]]


def answered_as(value, chain: list[dict], supplied: list[dict] = (), example_ids: set[str] = frozenset()) -> str:
    """What the answered value is, relative to the task's chain (and any supplied distractors)."""
    if value is None:
        return "unknown"
    entities = [chain[0]["subject"]] + [r["object"] for r in chain]
    if value in entities:
        return HOPS[entities.index(value)]
    if value in {r["id"] for r in chain}:
        return "chain_record_id"
    if value in {v for r in supplied for v in (r["subject"], r["object"])}:
        return "distractor_entity"
    if value in {r["id"] for r in supplied}:
        return "distractor_record_id"
    if value in example_ids:
        return "example_identifier"
    return "other"


def chain_depth(trace: dict, chain: list[dict]) -> int:
    """How many chain subjects were searched, in order from the start (0-4)."""
    searched = [s["action"].get("entity") for s in trace["steps"]
                if s.get("action", {}).get("action") == "search"]
    depth = 0
    for record in chain:
        if record["subject"] in searched:
            depth += 1
        else:
            break
    return depth


def citation_profile(trace: dict, task: dict, example_ids: set[str] = frozenset()) -> str:
    result = trace["result"]
    if result.get("status") != "answered" or result.get("value") is None:
        return "none"
    cited = result.get("evidence", [])
    if not cited:
        return "no_citations"
    expected = set(task["evidence"])
    observed = {r["id"] for step in trace["steps"] for r in step.get("observation", {}).get("records", [])}
    observed |= {r["id"] for r in trace.get("supplied", {}).get("records", [])}
    if set(cited) == expected:
        return "exact"
    if expected <= set(cited) and set(cited) <= observed:
        return "complete_with_extra_observed"
    if set(cited) & example_ids:
        return "cites_example_ids"
    if not set(cited) <= observed:
        return "cites_unobserved"
    return "incomplete_observed"


def citation_token(token: str, task: dict, record_ids: set[str], regions: set[str], entities: set[str],
                   example_ids: set[str]) -> str:
    if token in task["evidence"]:
        return "expected_record"
    if token in record_ids:
        return "other_record"
    if token in regions:
        return "region_entity"
    if token in entities:
        return "other_entity"
    if token in example_ids:
        return "example_identifier"
    return "other"


def analyze(world: dict, traces: list[dict], opening: list[dict] | None = None) -> dict:
    tasks = {t["id"]: t for t in world["tasks"]}
    example_ids = opening_identifiers(opening)
    record_ids = {r["id"] for r in world["records"]}
    regions = {r["object"] for r in world["records"] if r["relation"] == "located_in"}
    entities = {v for r in world["records"] for v in (r["subject"], r["object"])}
    statuses, answers, depths, citations, tokens, first_actions = (Counter() for _ in range(6))
    for trace in traces:
        task = tasks[trace["task_id"]]
        chain = chain_of(task, world["records"])
        result = trace["result"]
        statuses[result["status"]] += 1
        if result["status"] == "answered":
            supplied = trace.get("supplied", {}).get("records", [])
            answers[answered_as(result.get("value"), chain, supplied, example_ids)] += 1
            for token in result.get("evidence", []):
                tokens[citation_token(token, task, record_ids, regions, entities, example_ids)] += 1
        depths[chain_depth(trace, chain)] += 1
        citations[citation_profile(trace, task, example_ids)] += 1
        first = trace["steps"][0] if trace["steps"] else {}
        kind = first.get("action", {}).get("action", "invalid")
        if kind == "search":
            kind += "_start" if first["action"].get("entity") == task["start"] else "_other"
        first_actions[kind] += 1
    return {"tasks": len(traces), "statuses": dict(statuses), "answered_as": dict(answers),
            "chain_depth_searched": {str(k): v for k, v in sorted(depths.items())},
            "citations": dict(citations), "citation_tokens": dict(tokens), "first_action": dict(first_actions)}
