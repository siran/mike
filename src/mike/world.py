"""Seeded synthetic worlds. Answer keys stay outside the agent's tools."""
from __future__ import annotations

import hashlib
import json
import random
from pathlib import Path


def generate(seed: int, count: int) -> dict:
    if count < 1:
        raise ValueError("count must be positive")
    rng = random.Random(seed)
    used = set()

    def name(prefix: str) -> str:
        while True:
            candidate = prefix + format(rng.getrandbits(48), "012X")
            if candidate not in used:
                used.add(candidate)
                return candidate

    records, tasks = [], []
    for i in range(count):
        obj, box, station, region, prop = (name(p) for p in "OBSRP")
        pairs = [(obj, "inside", box), (box, "belongs_to", station),
                 (station, "located_in", region), (region, "has_property", prop)]
        ids = []
        for subject, relation, target in pairs:
            record_id = "R" + format(rng.getrandbits(64), "016X")
            ids.append(record_id)
            records.append({"id": record_id, "subject": subject, "relation": relation, "object": target})
        tasks.append({"id": f"task-{i:04d}", "question":
            f"What property characterizes the region containing the station associated with object {obj}?",
            "start": obj, "answer": prop, "evidence": ids})
    # Unrelated atomic facts make the database nontrivial without giving the
    # lookup engine a route to the answer or adding ambiguous task evidence.
    for _ in range(2000):
        records.append({"id": "R" + format(rng.getrandbits(64), "016X"),
                        "subject": name("D"), "relation": "related_to", "object": name("N")})
    rng.shuffle(records)
    return {"schema": 1, "seed": seed, "records": records, "tasks": tasks}


def canonical_hash(value: dict) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def save(world: dict, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(world, indent=2) + "\n", encoding="utf-8")


def load(path: str | Path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))
