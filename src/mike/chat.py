"""MIK-e chat: the plain-text kernel loop. A reply line starting with "$ " is a command for the body."""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from .body import has_command, parse_reply

DEFAULT_MODEL = "HuggingFaceTB/SmolLM2-360M-Instruct"
SYSTEM_FILE = Path(__file__).with_name("system-prompt.txt")


def _turns(*pairs):
    return [{"role": role, "content": content} for role, content in pairs]


# Worked conversations given as prior chat turns: small SmolLM2 models used tools only in this form
# (mike1-benchmark dev log, probes p01/p02). Numbers, facts and commands differ from the tests.
NOTES = "~/notes/user.md"
NOTES_CONTENT = "The user's dog is called Rex.\nThe user's sister is called Ana.\nThe user's number is 7."
WRITE_AND_SHOW = _turns(  # an earlier conversation
    ("user", "hi! remember this number: 7"),
    ("assistant", f"I'll write it down.\n$ echo \"The user's number is 7.\" >> {NOTES}"),
    ("user", "output:\n(no output)"),
    ("assistant", f"OK, I wrote it in {NOTES}."),
    ("user", "where did you write it?"),
    ("assistant", f"In my notes, in the file {NOTES}."),
    ("user", "show me the file"),
    ("assistant", f"$ cat {NOTES}"),
    ("user", f"output:\n{NOTES_CONTENT}"),
    ("assistant", f"That is all of {NOTES}. The last line is the one I just wrote."),
)
RECALL_AND_ADD = _turns(  # a later conversation
    ("user", "what's my dog's name?"),
    ("assistant", f"Let me check my notes.\n$ cat {NOTES}"),
    ("user", f"output:\n{NOTES_CONTENT}"),
    ("assistant", "Your dog is called Rex."),
    ("user", "and what number did I ask you to remember?"),
    ("assistant", "You asked me to remember 7."),
    ("user", "if you add 5 to that number?"),
    ("assistant", 'I\'ll use my calculator.\n$ python3 -c "print(7+5)"'),
    ("user", "output:\n12"),
    ("assistant", "7 + 5 = 12."),
)
LOOK_UP = _turns(
    ("user", "what's the population of Iceland?"),
    ("assistant", "Let me look it up.\n$ ask-web population of Iceland"),
    ("user", "output:\nWikipedia, Demographics of Iceland:\nThe demographics of Iceland include population "
             "density, education level, health of the populace, economic status, religious affiliations and "
             "other aspects of the population.\nOther pages: Iceland; Settlement of Iceland\n\n"
             "Web (DuckDuckGo):\n1. Demographics of Iceland - Wikipedia: As of 2026, the Icelandic population "
             "is 394,324. In 2026, 77,226 residents (19.6%) were immigrants. "
             "(https://en.wikipedia.org/wiki/Demographics_of_Iceland)"),
    ("assistant", "Iceland has about 394,000 people (394,324 in 2026)."),
    ("user", "who wrote Don Quixote?"),
    ("assistant", "Let me look it up.\n$ ask-web Don Quixote author"),
    ("user", "output:\nWikipedia, Don Quixote:\nDon Quixote is a novel by Miguel de Cervantes, written in "
             "Early Modern Spanish and published in two parts in 1605 and 1615. It is considered a "
             "founding work of Western literature and the first modern novel.\nOther pages: Don Quixote "
             "(character); List of works influenced by Don Quixote\n\nWeb (DuckDuckGo):\n"
             "1. Miguel de Cervantes - Wikipedia: Miguel de Cervantes was a Spanish writer and the author "
             "of Don Quixote, the first modern novel. (https://en.wikipedia.org/wiki/Miguel_de_Cervantes)"),
    ("assistant", "Don Quixote was written by Miguel de Cervantes."),
    ("user", "what does the wc command do?"),
    ("assistant", "Let me read its manual.\n$ man wc | head -n 12"),
    ("user", "output:\nWC(1)                            User Commands                            WC(1)\n\n"
             "NAME\n       wc - print newline, word, and byte counts for each file\n\nSYNOPSIS\n"
             "       wc [OPTION]... [FILE]...\n       wc [OPTION]... --files0-from=F\n\nDESCRIPTION\n"
             "       Print  newline, word, and byte counts for each FILE, and a total line if\n"
             "       more than one FILE is specified.  A word is a nonempty sequence  of  non"),
    ("assistant", "wc counts the lines, words and bytes in files. For example, wc -l FILE prints how "
                  "many lines FILE has."),
)
EXAMPLES = [LOOK_UP, WRITE_AND_SHOW, RECALL_AND_ADD]  # order probed: traces/probes/v2-*.jsonl


def opening(system: str | None = None) -> list[dict]:
    system = SYSTEM_FILE.read_text(encoding="utf-8").strip() if system is None else system
    return [{"role": "system", "content": system}] + [m for example in EXAMPLES for m in example]


def sha256(value) -> str:
    return hashlib.sha256(json.dumps(value, ensure_ascii=False).encode()).hexdigest()


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


class Transcript:
    """<stamp>.jsonl holds every event raw; <stamp>.txt is the readable conversation."""
    def __init__(self, directory: str | Path):
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y-%m-%d-%H%M%S-%f")[:-3]
        self.raw, self.text = directory / f"{stamp}.jsonl", directory / f"{stamp}.txt"

    def event(self, kind: str, **fields):
        with self.raw.open("a", encoding="utf-8") as f:
            f.write(json.dumps({"type": kind, "time": now(), **fields}, ensure_ascii=False) + "\n")

    def lines(self, prefix: str, text: str):
        with self.text.open("a", encoding="utf-8") as f:
            for line in text.splitlines() or [""]:
                f.write(f"{prefix}{line}\n")


def visible(lines: list[str]) -> list[str]:
    return [line for line in lines if line.strip() and not line.strip().startswith("```")]


class Session:
    def __init__(self, model, body, transcript: Transcript | None = None, max_steps: int = 8,
                 context_tokens: int = 6000, show=None, system: str | None = None):
        self.model, self.body, self.transcript = model, body, transcript
        self.max_steps, self.context_tokens = max_steps, context_tokens
        self.show = show or (lambda kind, text: None)
        self.opening = opening(system)
        self.history: list[dict] = []
        self.turn_starts: list[int] = []
        self.dropped = 0
        self.record("session", model=getattr(model, "metadata", {}), opening=self.opening,
                    opening_sha256=sha256(self.opening), max_steps=max_steps, context_tokens=context_tokens,
                    command_timeout=body.timeout, max_output=body.max_output, user=body.user, home=body.home)
        if transcript:
            transcript.lines("", f"# conversation {now()}, {getattr(model, 'metadata', {}).get('model')}")

    def record(self, kind: str, **fields):
        if self.transcript:
            self.transcript.event(kind, **fields)

    def window(self) -> list[dict]:
        """The opening plus as many recent whole turns as fit the token budget; the current turn stays."""
        while True:
            messages = self.opening + self.history[self.dropped:]
            if self.model.count_tokens(messages) <= self.context_tokens:
                return messages
            later = [i for i in self.turn_starts if i > self.dropped]
            if not later:
                return messages
            self.dropped = later[0]
            self.record("window", dropped_messages=self.dropped)

    def _add(self, role: str, content: str, prefix: str):
        self.history.append({"role": role, "content": content})
        if self.transcript:
            self.transcript.lines(prefix, content)

    def turn(self, text: str, max_new_tokens: int | None = None) -> dict:
        start = time.perf_counter()
        self.turn_starts.append(len(self.history))
        self._add("user", text, "user: ")
        self.record("user", text=text)
        report = {"user": text, "said": [], "commands": [], "model_calls": [], "seconds": 0.0}
        for step in range(self.max_steps + 1):
            messages = self.window()
            kwargs = {"max_new_tokens": max_new_tokens} if max_new_tokens else {}
            generation = self.model.generate(messages, stop=has_command, **kwargs)
            said, command, kept = parse_reply(generation["raw"])
            self._add("assistant", kept, "mike: ")
            self.record("model", raw=generation["raw"], kept=kept, said=said, command=command,
                        window_start=self.dropped, input_tokens=generation["input_tokens"],
                        output_tokens=generation["output_tokens"], reused_tokens=generation.get("reused_tokens"),
                        seconds=round(generation["inference_seconds"], 3))
            report["model_calls"].append({"seconds": generation["inference_seconds"],
                                          "input_tokens": generation["input_tokens"],
                                          "output_tokens": generation["output_tokens"]})
            speech = visible(said)
            report["said"] += speech
            for line in speech:
                self.show("say", line)
            if not speech and command is None:
                self.show("say", "(no reply)")
            self.show("time", f"{generation['inference_seconds']:.1f} s, {generation['input_tokens']} tokens in "
                              f"({generation.get('reused_tokens') or 0} reused), {generation['output_tokens']} out")
            if command is None:
                break
            if step == self.max_steps:
                note = f"output:\n(not run: the limit is {self.max_steps} commands per turn)"
                self._add("user", note, "")
                self.record("limit", command=command)
                self.show("output", note)
                break
            self.show("command", command)
            result = self.body.run(command)
            observation = self.body.observation(result)
            if step == self.max_steps - 1:
                observation += "\n(that was your last command for this turn)"
            self._add("user", observation, "")
            self.record("command", **result)
            report["commands"].append(command)
            self.show("output", observation)
        report["seconds"] = time.perf_counter() - start
        self.record("turn_end", seconds=round(report["seconds"], 3))
        return report

    def close(self):
        self.record("end")


# ---------- scripted conversations ----------
# One user turn per line. "#" comments, "/new" starts a new session, and after a turn:
#   expect: A | B           some spoken line of that turn contains A or B (case-insensitive)
#   expect-start: A | B     the turn's first spoken line starts with A or B
#   expect-command: A | B   some command run in that turn contains A or B
#   if-asked: TEXT          a user turn played only if the last turn's speech had a "?"
# "max-new-tokens: N" (or "default") applies to the turns that follow.

CHECKS = ("expect", "expect-start", "expect-command")


def read_script(text: str) -> list[tuple]:
    items = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line == "/new":
            items.append(("new",))
            continue
        key, sep, value = line.partition(":")
        if sep and key in CHECKS:
            items.append((key, [a.strip() for a in value.split("|") if a.strip()]))
        elif sep and key == "max-new-tokens":
            items.append(("max-new-tokens", None if value.strip() == "default" else int(value)))
        elif sep and key == "if-asked":
            items.append(("if-asked", value.strip()))
        else:
            items.append(("say", line))
    return items


def check(kind: str, alternatives: list[str], report: dict) -> bool:
    lows = [a.lower() for a in alternatives]
    if kind == "expect":
        return any(a in line.lower() for line in report["said"] for a in lows)
    if kind == "expect-start":
        return bool(report["said"]) and any(report["said"][0].strip().lower().startswith(a) for a in lows)
    return any(a in command.lower() for command in report["commands"] for a in lows)


def converse(new_session, items: list[tuple], out=print) -> dict:
    session, sessions, tokens = new_session(), 1, None
    turns, checks = [], []
    for item in items:
        if item[0] == "new":
            session.close()
            session, sessions = new_session(), sessions + 1
            out("--- new session ---")
        elif item[0] == "max-new-tokens":
            tokens = item[1]
        elif item[0] == "if-asked" and not (turns and any("?" in line for line in turns[-1]["said"])):
            out(f"(not asked, so not sent: {item[1]})")
        elif item[0] in ("say", "if-asked"):
            out(f"user> {item[1]}")
            turns.append(session.turn(item[1], max_new_tokens=tokens))
        elif not turns:
            raise ValueError(f"{item[0]} before any turn")
        else:
            checks.append({"turn": len(turns), "kind": item[0], "want": item[1],
                           "passed": check(item[0], item[1], turns[-1])})
    session.close()
    out("\n=== summary ===")
    for number, turn in enumerate(turns, 1):
        calls = turn["model_calls"]
        out(f"turn {number}: {turn['seconds']:.1f} s, {len(calls)} model call(s), "
            f"{sum(c['output_tokens'] for c in calls)} tokens out, commands: "
            f"{' ; '.join(turn['commands']) or '(none)'}")
    for c in checks:
        out(f"{'PASS' if c['passed'] else 'FAIL'}  turn {c['turn']} {c['kind']}: {' | '.join(c['want'])}")
    passed = sum(c["passed"] for c in checks)
    out(f"{passed}/{len(checks)} checks passed; {sessions} session(s); "
        f"{sum(t['seconds'] for t in turns):.1f} s over {len(turns)} turns")
    return {"turns": turns, "checks": checks, "passed": passed}


def printer(kind: str, text: str):
    if kind == "say":
        print(f"mike> {text}", flush=True)
    elif kind == "command":
        print(f"  $ {text}", flush=True)
    elif kind == "output":
        print("\n".join("  | " + line for line in text.splitlines()), flush=True)
    elif kind == "time":
        print(f"  ({text})", flush=True)


def chat(new_session):
    session = new_session()
    print("MIK-e. /new starts a new conversation, /quit leaves.", flush=True)
    while True:
        try:
            text = input("you> ").strip()
        except EOFError:
            break
        if not text:
            continue
        if text == "/quit":
            break
        if text == "/new":
            session.close()
            session = new_session()
            continue
        try:
            session.turn(text)
        except KeyboardInterrupt:
            session.record("interrupted")
            print("(interrupted)")
    session.close()
