from __future__ import annotations

import argparse
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from .evaluate import compare_runs, evaluate_one, summarize, symbolic_baseline
from .models import TransformersModel
from .runtime import run
from .tools import Tools
from .world import canonical_hash, generate, load, save


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def main():
    parser = argparse.ArgumentParser(prog="mike")
    subs = parser.add_subparsers(dest="command", required=True)
    gen = subs.add_parser("generate", help="Generate a private synthetic world and answer key")
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--count", type=int, default=100)
    gen.add_argument("--out", required=True)
    for name in ("ask", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--world", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--revision", default=None)
        p.add_argument("--max-steps", type=int, default=6)
        p.add_argument("--max-new-tokens", type=int, default=96)
        p.add_argument("--out", required=True)
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--no-tools", action="store_true")
        else:
            p.add_argument("--condition", choices=["tools", "no-tools"], required=True)
            p.add_argument("--limit", type=int, default=None)
    baseline = subs.add_parser("baseline")
    baseline.add_argument("--world", required=True)
    baseline.add_argument("--condition", choices=["tools", "no-tools"], required=True)
    compare = subs.add_parser("compare", help="Compare persisted evaluation traces by task ID")
    compare.add_argument("paths", nargs="+")
    compare.add_argument("--out", required=True)
    args = parser.parse_args()
    if args.command == "compare":
        runs = []
        for name in args.paths:
            with Path(name).open(encoding="utf-8") as f:
                lines = [json.loads(line) for line in f if line.strip()]
            metadata = next(item for item in lines if item.get("type") == "metadata")
            traces = [item for item in lines if item.get("type") == "trace"]
            runs.append((name, metadata, traces, any(item.get("type") == "summary" for item in lines)))
        comparison = compare_runs(runs)
        write_json(args.out, comparison)
        print(json.dumps(comparison))
        return
    if args.command == "generate":
        world = generate(args.seed, args.count)
        save(world, args.out)
        print(json.dumps({"path": args.out, "sha256": canonical_hash(world), "tasks": len(world["tasks"])}))
        return
    world = load(args.world)
    enabled = args.condition == "tools" if args.command != "ask" else not args.no_tools
    if args.command == "baseline":
        answers = [symbolic_baseline(t, world["records"], enabled) for t in world["tasks"]]
        correct = sum(a["value"] == t["answer"] for a, t in zip(answers, world["tasks"]))
        print(json.dumps({"baseline": "handwritten_four_relation_traversal", "condition": args.condition,
                          "tasks": len(answers), "correct": correct}))
        return
    model = TransformersModel(args.model, args.revision, args.max_new_tokens)
    metadata = {"model": model.metadata, "world_sha256": canonical_hash(world),
                "time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "max_steps": args.max_steps,
                "max_new_tokens": args.max_new_tokens, "tools_enabled": enabled}
    if args.command == "ask":
        trace = run(model, args.question, Tools(world["records"], enabled), args.max_steps)
        write_json(args.out, {"metadata": metadata, "trace": trace})
        print(json.dumps(trace["result"]))
        return
    if args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    tasks = world["tasks"][:args.limit]
    output = Path(args.out)
    output.parent.mkdir(parents=True, exist_ok=True)
    traces = []
    # Flush each task so a long CPU run remains inspectable if interrupted.
    with output.open("w", encoding="utf-8") as f:
        f.write(json.dumps({"type": "metadata", **metadata}) + "\n")
        f.flush()
        for task in tasks:
            trace = evaluate_one(model, task, world["records"], enabled, args.max_steps)
            traces.append(trace)
            f.write(json.dumps({"type": "trace", **trace}) + "\n")
            f.flush()
            print(f"{len(traces)}/{len(tasks)} {task['id']}: {trace['result']['status']} correct={trace['score']['correct']}", flush=True)
        f.write(json.dumps({"type": "summary", **summarize(traces)}) + "\n")
    print(json.dumps(summarize(traces)))


if __name__ == "__main__":
    main()
