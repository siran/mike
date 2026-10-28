from __future__ import annotations

import argparse
import json
import os
import platform
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from .body import Body
from .chat import DEFAULT_MODEL, Session, Transcript, chat, converse, printer, read_script
from .evaluate import compare_runs, evaluate_one, summarize, symbolic_baseline
from .models import TransformersModel
from .runtime import run
from .tools import Tools
from .world import canonical_hash, generate, load, save


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8", newline="\n")


def default_user():
    """As root (the container's kernel), commands run as `mike`; otherwise as the current user."""
    if os.environ.get("MIKE_USER"):
        return os.environ["MIKE_USER"]
    if hasattr(os, "geteuid") and os.geteuid() == 0:
        import pwd
        try:
            pwd.getpwnam("mike")
            return "mike"
        except KeyError:
            pass
    return None


def talk(args):
    script = None
    if args.command == "converse":
        script = read_script(sys.stdin.read() if args.script == "-" else Path(args.script).read_text(encoding="utf-8"))
    body = Body(args.home, args.user, args.timeout, args.max_output)
    conversations = Path(args.conversations or Path(body.home) / "conversations")
    print(f"loading {args.model} ...", flush=True)
    start = time.perf_counter()
    if os.environ.get("MIKE_SERVER"):  # a candidate kernel served by llama.cpp (exam/run.sh)
        from mike.server_model import ServerModel
        model = ServerModel(os.environ["MIKE_SERVER"], int(os.environ.get("MIKE_MAX_NEW_TOKENS") or 1024))
    else:
        model = TransformersModel(args.model, args.revision, args.max_new_tokens, args.threads,
                                  reuse_cache=not args.no_reuse_cache, repetition_penalty=args.repetition_penalty,
                                  no_repeat_ngram_size=args.no_repeat_ngram_size)
    print(f"loaded in {time.perf_counter() - start:.1f} s; {model.metadata['parameters']:,} parameters, "
          f"{model.metadata['torch_threads']} threads; commands run as {body.user or 'the current user'} "
          f"in {body.home}; transcripts in {conversations}", flush=True)

    def new_session():
        return Session(model, body, Transcript(conversations), args.max_steps, args.context_tokens, printer)

    if script is None:
        chat(new_session)
        return 0
    result = converse(new_session, script)
    return 0 if result["passed"] == len(result["checks"]) else 1


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
    for name, text in (("chat", "Talk with MIK-e"), ("converse", "Play a scripted conversation and report")):
        p = subs.add_parser(name, help=text)
        p.add_argument("--model", default=os.environ.get("MIKE_MODEL") or DEFAULT_MODEL)
        p.add_argument("--revision", default=None)
        p.add_argument("--user", default=default_user(), help="run commands as this user")
        p.add_argument("--home", default=None, help="working directory for commands (default: the user's home)")
        p.add_argument("--conversations", default=None, help="transcript directory (default: HOME/conversations)")
        p.add_argument("--max-steps", type=int, default=8, help="commands per user turn")
        p.add_argument("--timeout", type=float, default=30.0, help="seconds per command")
        p.add_argument("--max-output", type=int, default=2000, help="characters of command output shown")
        p.add_argument("--max-new-tokens", type=int, default=200)
        p.add_argument("--context-tokens", type=int, default=6000)
        p.add_argument("--threads", type=int, default=int(os.environ.get("MIKE_THREADS", 0)) or None)
        p.add_argument("--repetition-penalty", type=float, default=1.0,
                       help="Hugging Face repetition_penalty; it also counts the prompt and worked examples")
        p.add_argument("--no-repeat-ngram-size", type=int, default=0,
                       help="Hugging Face no_repeat_ngram_size; it also counts the prompt and worked examples")
        p.add_argument("--no-reuse-cache", action="store_true",
                       help="recompute the whole prompt every call instead of reusing the shared prefix")
        if name == "converse":
            p.add_argument("--script", required=True, help="script file, or - for stdin")
    args = parser.parse_args()
    if args.command in ("chat", "converse"):
        return talk(args)
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
    sys.exit(main())
