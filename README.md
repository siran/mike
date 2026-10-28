# MIK-e — a minimal intelligence kernel, embodied

How small can a mind be, if it doesn't have to carry the world in its head?

Large language models store an enormous amount of the world in their weights. MIK-e asks the opposite
question: give a **small, frozen** model a *body* — a place to work, things to look up, notes it can keep,
a history it can return to — and see how little learned capacity is needed to solve tasks it has never
seen. The facts live outside; the kernel only has to understand a question, decide what to look for,
compare what it finds, and say what it concluded and why.

This repository is a research instrument, not a product. It measures a narrow thing carefully rather
than a broad thing loosely, and it reports negative results as plainly as positive ones.

## The idea in one picture

```text
            question ──►  ┌──────────────┐   "search OB23C…"   ┌───────────────┐
                          │    kernel    │ ──────────────────► │     body      │
                          │ (small, fro- │                     │ sources, desk,│
                          │  zen model)  │ ◄────────────────── │ notes, history│
            answer  ◄──   └──────────────┘   "OB23C… is in …"  └───────────────┘
           + evidence
```

* **The kernel** is an off-the-shelf small language model (so far SmolLM2-135M and -360M), never
  fine-tuned. It decides one action at a time.
* **The body** executes what the kernel asks — look up a subject, read or write a note, report an answer —
  and returns what it observed. It never chooses the query, plans the chain, or picks the answer: anything
  that does is counted as part of the system's intelligence, not hidden in the harness.
* **The world** is generated: random entities, random facts, random identifiers. A model cannot have
  memorised them, so a correct answer has to come from what it looked up.

The body is inspired by [eGPT](https://github.com/siran/egpt), where each conversation has a durable
desk, a transcript, and drawers of material that outlive any single model invocation. See
[MIK-e's body](specs/body.md) for the design sketch and what it does *not* yet claim.

## Status

**MIK-0 (2026-10-28): a negative result about the interface, not about reasoning.**
Both models scored 0/100 on a four-hop retrieval task, with and without search — because neither ever
issued a search. The 360M model answered immediately with the question's own identifier; the 135M model
mostly echoed the JSON it was given. Since no retrieved evidence ever reached either model, the run says
nothing yet about whether a small kernel can combine facts. A handwritten four-step traversal scored
100/100 with the lookup and 0/100 without it, so the task is solvable from the body.
Full report: [reports/mik0-2026-10-28.md](reports/mik0-2026-10-28.md).

**Next — MIK-e-1: talk to the kernel in plain text.** MIK-0 fed the model a JSON state and demanded a
JSON action on its first turn. A minimal kernel shouldn't need that: it should read and write ordinary
language. MIK-e-1 develops a plain-text interface on a separate development world (a worked example,
observations as sentences, short text commands for actions), measures evidence-supplied reasoning
separately from tool use, then freezes the protocol and evaluates on a fresh held-out world.

## How it is measured

Before interpreting any number, read:

* [hypothesis.md](specs/hypothesis.md) — what is and isn't being tested;
* [protocol.md](specs/protocol.md) — the pre-registered MIK-0 procedure;
* [evaluation.md](specs/evaluation.md) — scoring, evidence support, and cost proxies.

Principles the instrument holds to:

* **Pre-registration.** A protocol is written and frozen before its held-out run; development happens on
  separate worlds.
* **No silent repair.** An invalid action is a recorded failure, never quietly fixed.
* **Evidence, not just answers.** A correct answer counts as supported only if it cites the records it
  actually observed.
* **Costs next to scores.** Parameters, tokens, lookups, wall and CPU time are reported with every result;
  no efficiency claim follows from parameter count alone.
* **Everything retained.** Every trace, error and interrupted run is kept and identified.

## Install

Python 3.10+ and a CPU are enough.

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.\.venv\Scripts\python -m pip install -e '.[models]'
```

The first use of a model downloads its weights from Hugging Face; the resolved snapshot is recorded in
every trace (`--revision` pins one). Weights are never modified.

## Run

```powershell
mike generate --seed 20261029 --count 100 --out data/heldout.json
mike baseline --world data/heldout.json --condition tools
mike evaluate --world data/heldout.json --model HuggingFaceTB/SmolLM2-135M-Instruct --condition tools    --out traces/135m-tools.jsonl
mike evaluate --world data/heldout.json --model HuggingFaceTB/SmolLM2-135M-Instruct --condition no-tools --out traces/135m-no-tools.jsonl
mike compare traces/135m-tools.jsonl traces/135m-no-tools.jsonl --out reports/comparison.json
mike ask --world data/heldout.json --model MODEL --out traces/ask.json "Your question"
```

Use `--limit 1` for a smoke test and a different world seed for development than for held-out work.
Search understands exact subject identifiers from the generated world; it is not a web search.

Tests: `python -m unittest discover -s tests -v`.

## Layout

```text
src/mike/      runtime (the bounded kernel–body loop), models, tools, world generator, evaluation, CLI
specs/         hypothesis, protocol, measurement, and the body design
reports/       written results, with machine-readable comparisons
tests/         instrument tests
data/, traces/ generated worlds and per-step traces (local, not committed)
```

## Open questions

Rule discovery, contradictory or misleading evidence, composing tasks, persistent notes across
questions, calibration, multiple independent worlds, stronger reference models, and full system-cost
accounting are all future work. The point of starting small is to know exactly what each step shows —
and what it doesn't.
