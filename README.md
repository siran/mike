# MIK — Minimal Intelligence Kernel

An early research instrument for testing whether small frozen language models can use externally retrieved information to solve unfamiliar tasks. It does not claim to measure intelligence in general or find a minimum architecture. See [hypothesis](specs/hypothesis.md), [protocol](specs/protocol.md), and [measurement](specs/evaluation.md) before interpreting results.

## Install

Python 3.10+ and CPU inference are sufficient. Install PyTorch using the appropriate instructions for your platform, then install this project and its model dependencies:

```powershell
py -m venv .venv
.\.venv\Scripts\python -m pip install --index-url https://download.pytorch.org/whl/cpu torch
.\.venv\Scripts\python -m pip install -e '.[models]'
```

The first use of a model downloads its weights from Hugging Face. Model snapshot identifiers are included in traces; use `--revision` to pin one explicitly. No model weights are changed. `data/` and `traces/` are local and ignored by Git.

## Run

```powershell
mik generate --seed 20261029 --count 100 --out data/heldout.json
mik baseline --world data/heldout.json --condition tools
mik evaluate --world data/heldout.json --model HuggingFaceTB/SmolLM2-135M-Instruct --condition tools --out traces/135m-tools.jsonl
mik evaluate --world data/heldout.json --model HuggingFaceTB/SmolLM2-135M-Instruct --condition no-tools --out traces/135m-no-tools.jsonl
```

Repeat the evaluation for `HuggingFaceTB/SmolLM2-360M-Instruct`. Use `--limit 1` for a smoke test, and a separate world seed for smoke vs held-out work. `mik ask --world data/heldout.json --model MODEL --out traces/ask.json "Your question"` runs one bounded interaction. Search understands exact subject identifiers from the world; it is not a general web search. A deterministic `measure` action exists as an interface fixture, not yet as a discovery benchmark.

Use `mik compare traces/135m-tools.jsonl traces/135m-no-tools.jsonl --out reports/comparison.json` to summarize completed or interrupted runs and compute paired outcomes for tasks from the same world.

Run the instrument tests with `python -m unittest discover -s tests -v`. The proposal's rule discovery, contradiction, compositional generalization, persistent memory, and training experiments remain future work. Local traces may contain questions and observations; review them before sharing.
