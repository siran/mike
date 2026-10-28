"""Narrow deterministic tools: no model, graph traversal, or answer key."""
from __future__ import annotations


class Tools:
    def __init__(self, records: list[dict], enabled: bool = True):
        self.enabled = enabled
        self.index = {}
        for record in records:
            self.index.setdefault(record["subject"], []).append(record)

    def execute(self, action: dict) -> dict:
        kind = action.get("action")
        if not self.enabled:
            return {"error": "tools_disabled"}
        if kind == "search":
            entity = action.get("entity")
            if not isinstance(entity, str) or len(entity) > 100:
                return {"error": "entity_must_be_a_string"}
            return {"records": self.index.get(entity, [])}
        if kind == "measure":
            # An isolated experiment tool; not used by the retrieval evaluation.
            x = action.get("x")
            if isinstance(x, bool) or not isinstance(x, int) or abs(x) > 1000:
                return {"error": "x_must_be_an_integer_between_-1000_and_1000"}
            return {"x": x, "y": 3 * x + 1}
        return {"error": "unknown_action"}
