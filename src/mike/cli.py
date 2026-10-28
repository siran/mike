from __future__ import annotations

import argparse
import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path

from . import text
from .analysis import analyze
from .evaluate import compare_runs, evaluate_one, evidence_baselines, summarize, symbolic_baseline
from .models import SYSTEM, TransformersModel
from .runtime import run, run_text
from .tools import Tools
from .world import canonical_hash, generate, load, save


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def source_sha256():
    digest = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        digest.update(path.name.encode() + b"\0" + path.read_bytes())
    return digest.hexdigest()


def interface_metadata(interface, condition, retry, world):
    if interface == "json":
        return {"interface": "json", "condition": condition, "retry": False,
                "system_prompt": SYSTEM, "opening_sha256": text.sha256(SYSTEM), "opening": None,
                "early_stop": None}
    if not text.examples_disjoint(world):
        raise SystemExit("worked-example identifiers occur in this world; refusing to run")
    opening = text.opening(interface, condition)
    return {"interface": interface, "condition": condition, "retry": retry,
            "system_prompt": opening[0]["content"], "opening": opening,
            "opening_sha256": text.sha256(json.dumps(opening, ensure_ascii=False)),
            "retry_message": text.RETRY_MESSAGE if retry else None,
            "early_stop": "stop once a newline-terminated generated line parses as a command",
            "distractors_per_task": text.DISTRACTORS_PER_TASK if condition == "evidence" else None}


def main():
    parser = argparse.ArgumentParser(prog="mike")
    subs = parser.add_subparsers(dest="command", required=True)
    gen = subs.add_parser("generate", help="Generate a private synthetic world and answer key")
    gen.add_argument("--seed", type=int, required=True)
    gen.add_argument("--count", type=int, default=100)
    gen.add_argument("--schema", type=int, choices=[1, 2], default=2,
                     help="1: MIK-0 worlds (record IDs start with R); 2: record IDs start with F")
    gen.add_argument("--out", required=True)
    for name in ("ask", "evaluate"):
        p = subs.add_parser(name)
        p.add_argument("--world", required=True)
        p.add_argument("--model", required=True)
        p.add_argument("--revision", default=None)
        p.add_argument("--max-steps", type=int, default=6)
        p.add_argument("--max-new-tokens", type=int, default=96)
        p.add_argument("--threads", type=int, default=None, help="torch.set_num_threads")
        p.add_argument("--interface", choices=text.INTERFACES, default="json")
        p.add_argument("--retry", action="store_true",
                       help="text interfaces: one explicit retry message per task after an invalid output")
        p.add_argument("--out", required=True)
        if name == "ask":
            p.add_argument("question")
            p.add_argument("--no-tools", action="store_true")
        else:
            p.add_argument("--condition", choices=["tools", "no-tools", "evidence"], required=True)
            p.add_argument("--limit", type=int, default=None)
    baseline = subs.add_parser("baseline")
    baseline.add_argument("--world", required=True)
    baseline.add_argument("--condition", choices=["tools", "no-tools", "evidence"], required=True)
    compare = subs.add_parser("compare", help="Compare persisted evaluation traces by task ID")
    compare.add_argument("paths", nargs="+")
    compare.add_argument("--world", default=None, help="add failure analysis against this world's answer key")
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
        if args.world:
            world = load(args.world)
            for entry, (_, metadata, traces, _) in zip(comparison["runs"], runs):
                if metadata["world_sha256"] != canonical_hash(world):
                    raise SystemExit(f"{entry['label']} was not run on {args.world}")
                entry["analysis"] = analyze(world, traces, metadata.get("opening"))
        write_json(args.out, comparison)
        print(json.dumps(comparison))
        return
    if args.command == "generate":
        world = generate(args.seed, args.count, args.schema)
        save(world, args.out)
        print(json.dumps({"path": args.out, "sha256": canonical_hash(world), "tasks": len(world["tasks"])}))
        return
    world = load(args.world)
    if args.command == "baseline":
        if args.condition == "evidence":
            print(json.dumps({"baseline": "evidence_supplied_controls", **evidence_baselines(world)}))
            return
        enabled = args.condition == "tools"
        answers = [symbolic_baseline(t, world["records"], enabled) for t in world["tasks"]]
        correct = sum(a["value"] == t["answer"] for a, t in zip(answers, world["tasks"]))
        print(json.dumps({"baseline": "handwritten_four_relation_traversal", "condition": args.condition,
                          "tasks": len(answers), "correct": correct}))
        return
    condition = args.condition if args.command == "evaluate" else ("no-tools" if args.no_tools else "tools")
    if args.interface == "json" and (condition == "evidence" or args.retry):
        parser.error("the JSON interface supports only the tools and no-tools conditions, without retry")
    enabled = condition == "tools"
    details = interface_metadata(args.interface, condition, args.retry, world)
    model = TransformersModel(args.model, args.revision, args.max_new_tokens, args.threads)
    metadata = {"model": model.metadata, "world_sha256": canonical_hash(world), "world_seed": world.get("seed"),
                "time_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
                "platform": platform.platform(), "source_sha256": source_sha256(), "max_steps": args.max_steps,
                "max_new_tokens": args.max_new_tokens, "tools_enabled": enabled, **details}
    if args.command == "ask":
        tools = Tools(world["records"], enabled)
        if args.interface == "json":
            trace = run(model, args.question, tools, args.max_steps)
        else:
            trace = run_text(model, args.question, tools, args.max_steps,
                             opening=details["opening"], retry=args.retry)
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
            facts = text.supplied_facts(world, task) if condition == "evidence" else None
            trace = evaluate_one(model, task, world["records"], enabled, args.max_steps,
                                 interface=args.interface, opening=details["opening"],
                                 facts=facts, retry=args.retry)
            traces.append(trace)
            f.write(json.dumps({"type": "trace", **trace}) + "\n")
            f.flush()
            print(f"{len(traces)}/{len(tasks)} {task['id']}: {trace['result']['status']} "
                  f"correct={trace['score']['correct']} steps={len(trace['steps'])}", flush=True)
        f.write(json.dumps({"type": "summary", **summarize(traces)}) + "\n")
    print(json.dumps(summarize(traces)))


if __name__ == "__main__":
    main()
