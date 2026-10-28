"""Plain-text kernel interface: English prompts, one-line commands, sentence observations.

The parser accepts a declared grammar and nothing else; it never repairs output.
"""
from __future__ import annotations

import hashlib
import random
import re

INTERFACES = ("json", "text", "text-example")

SYSTEM_TOOLS = """You answer questions about a database of facts. You do not know these facts; you have to look them up.

To look up an ID, reply with one line:
search <ID>
I will reply with the facts about that ID. Each fact starts with its record ID.

Look up one ID at a time. Follow the IDs that the facts give you until you reach what the question asks for.

When you know the answer, reply with one line:
answer <ID> because <record IDs>
List the record IDs of the facts you used. If you cannot find the answer, reply: answer unknown"""

SYSTEM_EVIDENCE = """You answer questions about a database of facts. You do not know these facts yourself. The facts you need are listed before the question, mixed with some facts you do not need. Each fact starts with its record ID.

Follow the facts from the ID in the question until you reach what the question asks for.

Reply with one line:
answer <ID> because <record IDs>
List the record IDs of the facts you used. If the facts do not give the answer, reply: answer unknown"""

# Made-up identifiers from random.Random("mike1-example-ids-not-from-any-world");
# examples_disjoint() checks them against each evaluated world.
_EXAMPLE_CHAIN = [
    {"id": "F5F0943517C5BF050", "subject": "OC466B41C0A7A", "relation": "inside", "object": "B17EAFA897FE4"},
    {"id": "FBFFC342D72A3C297", "subject": "B17EAFA897FE4", "relation": "belongs_to", "object": "S0A547DC6F7DA"},
    {"id": "F33983177FAC87BB6", "subject": "S0A547DC6F7DA", "relation": "located_in", "object": "R11ABC5B76157"},
    {"id": "F02513CB5017CC1E0", "subject": "R11ABC5B76157", "relation": "has_property", "object": "PB716400BF4D4"},
]
_EXAMPLE_DISTRACTORS = [
    {"id": "F76BF518358361E97", "subject": "O16C8FE1A6338", "relation": "inside", "object": "BBACD99B8B147"},
    {"id": "FB79AA9D1B393FB8B", "subject": "B578D6F546FB7", "relation": "belongs_to", "object": "S5DC803D60D28"},
    {"id": "FA74697F5BB24B6B4", "subject": "S57805880119F", "relation": "located_in", "object": "RB54135AD1669"},
    {"id": "F1046718BAB294F68", "subject": "R521B2CBC7D1D", "relation": "has_property", "object": "PA5CBA8559508"},
]
_EXAMPLE_QUESTION = ("What property characterizes the region containing the station associated with object "
                     "OC466B41C0A7A?")

RETRY_MESSAGE = "I didn't understand that. Reply with a line starting with search or answer."

ENTITY_TYPES = {"O": "object", "B": "box", "S": "station", "R": "region", "P": "property"}
ENTITY_ID_LENGTH = 13
RELATION_PHRASES = {"inside": "is inside", "belongs_to": "belongs to", "located_in": "is located in",
                    "has_property": "has", "related_to": "is related to"}

DISTRACTORS_PER_TASK = 4
# Development switches, both off in the frozen interface (specs/dev-log-mike1.md, protocol-mike1.md).
TYPE_NOUNS = False
REMIND_QUESTION = False
CHAIN_RELATIONS = ("inside", "belongs_to", "located_in", "has_property")


# ---------- rendering ----------

def entity(identifier: str) -> str:
    """An entity with the type its generated identifier encodes (world.generate's O/B/S/R/P prefixes)."""
    noun = ENTITY_TYPES.get(identifier[:1]) if TYPE_NOUNS and len(identifier) == ENTITY_ID_LENGTH else None
    return f"{noun} {identifier}" if noun else identifier


def render_record(record: dict) -> str:
    phrase = RELATION_PHRASES.get(record["relation"], record["relation"].replace("_", " "))
    return f"Record {record['id']}: {entity(record['subject'])} {phrase} {entity(record['object'])}."


def render_observation(entity: str, observation: dict) -> str:
    if observation.get("error") == "tools_disabled":
        return "Search is disabled. No facts are available."
    if "error" in observation:
        return f"Search failed: {observation['error'].replace('_', ' ')}."
    records = observation.get("records", [])
    if not records:
        return f"There are no facts about {entity}."
    return "\n".join([f"Facts about {entity}:"] + [render_record(r) for r in records])


def observation_message(searched: str, observation: dict, question: str) -> str:
    rendered = render_observation(searched, observation)
    return f"{rendered}\n\nQuestion: {question}" if REMIND_QUESTION else rendered


def render_facts(records: list[dict]) -> str:
    return "\n".join(["Facts:"] + [render_record(r) for r in records])


def first_user_message(question: str, facts: list[dict] | None = None) -> str:
    if facts is None:
        return question
    return f"{render_facts(facts)}\n\nQuestion: {question}"


def _answer_line(chain):
    return f"answer {chain[-1]['object']} because " + ", ".join(r["id"] for r in chain)


def _example_turns(evidence: bool) -> list[dict]:
    """One worked example trajectory as prior chat turns, with made-up IDs."""
    chain = _EXAMPLE_CHAIN
    if evidence:
        facts = [_EXAMPLE_DISTRACTORS[2], chain[1], _EXAMPLE_DISTRACTORS[0], chain[3],
                 chain[0], _EXAMPLE_DISTRACTORS[3], chain[2], _EXAMPLE_DISTRACTORS[1]]
        turns = [{"role": "user", "content": first_user_message(_EXAMPLE_QUESTION, facts)}]
    else:
        turns = [{"role": "user", "content": first_user_message(_EXAMPLE_QUESTION)}]
        for record in chain:
            turns += [{"role": "assistant", "content": f"search {record['subject']}"},
                      {"role": "user", "content": observation_message(record["subject"], {"records": [record]},
                                                                       _EXAMPLE_QUESTION)}]
    return turns + [{"role": "assistant", "content": _answer_line(chain)}]


def system_prompt(interface: str, condition: str) -> str:
    if interface not in ("text", "text-example"):
        raise ValueError(f"not a text interface: {interface}")
    return SYSTEM_EVIDENCE if condition == "evidence" else SYSTEM_TOOLS


def opening(interface: str, condition: str) -> list[dict]:
    """Messages before the task's first user turn: the system prompt, then any worked example."""
    messages = [{"role": "system", "content": system_prompt(interface, condition)}]
    if interface == "text-example":
        messages += _example_turns(condition == "evidence")
    return messages


def example_identifiers() -> set[str]:
    return {v for r in _EXAMPLE_CHAIN + _EXAMPLE_DISTRACTORS for v in (r["id"], r["subject"], r["object"])}


def examples_disjoint(world: dict) -> bool:
    used = {v for r in world["records"] for v in (r["id"], r["subject"], r["object"])}
    return not (used & example_identifiers())


def sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


# ---------- parsing ----------
# Grammar, per line after stripping surrounding whitespace and at most one trailing period:
#   search <token>
#   answer <token> [because <token>[, <token>]...]
# Command words are case-insensitive and may be followed by a colon. Citations are separated by
# commas and/or spaces; the word "and" between them is a separator. "answer unknown" abstains.
# Every other token is kept exactly as written. The first line that matches is the action.

_SEARCH = re.compile(r"search:?\s+(\S+)", re.IGNORECASE)
_ANSWER = re.compile(r"answer:?\s+(\S+)(?:\s+because\s+(\S.*))?", re.IGNORECASE)


def parse_line(line: str) -> dict | None:
    line = line.strip()
    if line.endswith("."):
        line = line[:-1].rstrip()
    match = _SEARCH.fullmatch(line)
    if match:
        return {"action": "search", "entity": match.group(1)}
    match = _ANSWER.fullmatch(line)
    if match:
        value = match.group(1)
        evidence = [t for t in re.split(r"[,\s]+", match.group(2) or "") if t and t.lower() != "and"]
        return {"action": "answer", "value": None if value.lower() == "unknown" else value,
                "evidence": evidence}
    return None


def parse(raw: str) -> tuple[dict | None, str | None]:
    for line in raw.splitlines():
        action = parse_line(line)
        if action is not None:
            return action, line.strip()
    return None, None


def has_complete_command(generated: str) -> bool:
    """Early-stop test: a newline-terminated line already parses, so parse() cannot change."""
    if "\n" not in generated:
        return False
    return any(parse_line(line) is not None for line in generated[:generated.rfind("\n")].splitlines())


# ---------- evidence-supplied condition ----------

def supplied_facts(world: dict, task: dict, distractors: int = DISTRACTORS_PER_TASK) -> list[dict]:
    """The task's four records plus one record of each chain relation from four other tasks, shuffled."""
    by_id = {r["id"]: r for r in world["records"]}
    rng = random.Random(f"mike1-evidence:{world['seed']}:{task['id']}")
    others = [t for t in world["tasks"] if t["id"] != task["id"]]
    if len(others) < distractors:
        raise ValueError("not enough other tasks for distractors")
    donors = rng.sample(others, distractors)
    chosen = [by_id[donor["evidence"][i % len(CHAIN_RELATIONS)]] for i, donor in enumerate(donors)]
    facts = [by_id[i] for i in task["evidence"]] + chosen
    rng.shuffle(facts)
    return facts


def first_property_baseline(facts: list[dict]) -> str | None:
    """No-join control for the evidence condition: the first property listed."""
    return next((r["object"] for r in facts if r["relation"] == "has_property"), None)
