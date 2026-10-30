"""Remove one direction from a model's residual stream, in its weights (directional ablation, Arditi et al. 2024).

  find   activations for contrasting prompts -> candidate directions, each scored on held-out prompts
  apply  orthogonalize every residual-stream writer against the chosen direction; save a new model

Targets:  helpless  he should start computer questions with "$ "        (kernel/requests.txt)
          persona   he should not start questions about himself "As an"   (kernel/persona.txt)

Streams one decoder layer at a time from the safetensors files, so a 4B model needs about 3 GB of RAM.
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

import torch
import transformers
from safetensors import safe_open
from safetensors.torch import save_file
from transformers import AutoConfig, AutoTokenizer

WRITERS = ("self_attn.o_proj.weight", "self_attn.o_proj.bias", "mlp.down_proj.weight", "mlp.down_proj.bias")
EMBED = "model.embed_tokens.weight"
TARGETS = {
    "helpless": {"ask": "act", "other": "helpless system", "want": "$"},
    "persona": {"ask": "self", "other": "ai system", "avoid": "As an"},
}


def read_requests(path):
    sections, name = {}, None
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            name = line[1:-1]
            sections[name] = []
        else:
            sections[name].append(line)
    return sections


class Weights:
    def __init__(self, model_dir):
        self.dir = Path(model_dir)
        index = self.dir / "model.safetensors.index.json"
        if index.exists():
            self.where = json.loads(index.read_text())["weight_map"]
        else:
            with safe_open(self.dir / "model.safetensors", "pt") as f:
                self.where = {k: "model.safetensors" for k in f.keys()}

    def get(self, name):
        with safe_open(self.dir / self.where[name], "pt") as f:
            return f.get_tensor(name)

    def layer(self, i):
        prefix = f"model.layers.{i}."
        return {k[len(prefix):]: self.get(k) for k in self.where if k.startswith(prefix)}


def ablate(x, r):
    return x - (x @ r).unsqueeze(-1) * r


class Streamer:
    def __init__(self, model_dir):
        self.config = AutoConfig.from_pretrained(model_dir)
        self.config._attn_implementation = "sdpa"
        self.w = Weights(model_dir)
        module = sys.modules[getattr(transformers, self.config.architectures[0]).__module__]

        def find(suffix):
            return next(getattr(module, n) for n in dir(module) if n.endswith(suffix) and not n.startswith("_"))

        self.layer = find("DecoderLayer")(self.config, 0).float().eval()
        self.rotary = find("RotaryEmbedding")(config=self.config)
        self.norm = find("RMSNorm")(self.config.hidden_size, eps=self.config.rms_norm_eps).float()
        self.norm.load_state_dict({"weight": self.w.get("model.norm.weight")})
        self.embed = self.w.get(EMBED)
        self.tied = "lm_head.weight" not in self.w.where
        self.unembed = self.embed if self.tied else self.w.get("lm_head.weight")
        self.n_layers = self.config.num_hidden_layers

    def logits(self, x, r=None):
        u = self.unembed
        out = torch.cat([x @ u[a:a + 32768].float().T for a in range(0, u.shape[0], 32768)], dim=1)
        if r is not None and self.tied:  # the edited embedding is also the unembedding
            u_r = torch.cat([u[a:a + 32768].float() @ r for a in range(0, u.shape[0], 32768)])
            out = out - (x @ r).unsqueeze(-1) * u_r
        return out

    @torch.inference_mode()
    def run(self, prompts, r=None, collect=False, tails=None):
        """Logits at the last `tail` positions of each prompt (a list of [tail, vocab]);
        with collect, also the residual stream entering each layer at the last position."""
        tails = tails or [1] * len(prompts)
        hs = [self.embed[ids].float() for ids in prompts]
        if r is not None:
            hs = [ablate(h, r) for h in hs]
        pos = [(torch.arange(len(h))[None], self.rotary(h[None], torch.arange(len(h))[None])) for h in hs]
        hooks = [] if r is None else [m.register_forward_hook(lambda _m, _i, out: ablate(out, r))
                                      for m in (self.layer.self_attn.o_proj, self.layer.mlp.down_proj)]
        acts = []
        try:
            for i in range(self.n_layers):
                if collect:
                    acts.append(torch.stack([h[-1] for h in hs]))
                self.layer.load_state_dict(self.w.layer(i))
                nxt = []
                for h, (pid, pe) in zip(hs, pos):
                    out = self.layer(h[None], attention_mask=None, position_ids=pid, position_embeddings=pe)
                    nxt.append((out[0] if isinstance(out, tuple) else out)[0])
                hs = nxt
        finally:
            for hook in hooks:
                hook.remove()
        flat = self.logits(self.norm(torch.cat([h[-t:] for h, t in zip(hs, tails)])), r)
        logits = list(torch.split(flat, tails))
        return (logits, torch.stack(acts)) if collect else logits


def encode(tokenizer, system, user):
    ids = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                        add_generation_prompt=True, tokenize=True)
    return torch.tensor(ids if isinstance(ids, list) else ids["input_ids"])


def find(args):
    torch.set_num_threads(args.threads)
    target = TARGETS[args.target]
    ask = target["ask"]
    req = read_requests(args.requests)
    system = Path(args.system).read_text(encoding="utf-8").strip()
    other = " ".join(req[target["other"]])
    tok = AutoTokenizer.from_pretrained(args.model)
    s = Streamer(args.model)
    dollar = torch.tensor([i for i in range(len(tok)) if tok.decode([i]).lstrip().startswith("$")])
    if "want" in target:
        start, prefix = dollar, None
        label = target["want"]
    else:
        prefix = torch.tensor(tok.encode(target["avoid"], add_special_tokens=False))
        word = tok.decode(prefix[:1]).strip()
        start = torch.tensor([i for i in range(len(tok)) if tok.decode([i]).strip() == word])
        label = target["avoid"]
    print(f"{s.config.architectures[0]}, {s.n_layers} layers; target {args.target}: replies to '{ask}' "
          f"{'should' if prefix is None else 'should not'} start with {label!r}", flush=True)

    t = time.time()
    tr_other = [encode(tok, other, q) for q in req[f"{ask} train"]]
    tr_mike = [encode(tok, system, q) for q in req[f"{ask} train"]]
    tr_talk = [encode(tok, system, q) for q in req["talk train"]]
    train_logits, acts = s.run(tr_other + tr_mike + tr_talk, collect=True)
    n = len(tr_mike)
    a_other, a_mike, a_talk = acts[:, :n], acts[:, n:2 * n], acts[:, 2 * n:]
    designs = {"situation": a_other.mean(1) - a_mike.mean(1), "request": a_mike.mean(1) - a_talk.mean(1)}
    first = torch.stack([lg[0] for lg in train_logits[n:2 * n]])
    hit = torch.log_softmax(first, -1)[:, start].exp().sum(-1) > 0.5
    print(f"'{ask}' train, Mike's prompt: {int(hit.sum())}/{n} start with {label!r}; "
          f"activations in {time.time() - t:.0f} s", flush=True)
    if 3 <= int(hit.sum()) <= n - 3:  # his own replies, split by what he did
        designs["reflex"] = a_mike[:, hit].mean(1) - a_mike[:, ~hit].mean(1)

    dev_ask = [encode(tok, system, q) for q in req[f"{ask} dev"]]
    tail = 1
    if prefix is not None:  # teacher-force the avoided start, to read how likely it is
        dev_ask = [torch.cat([p, prefix]) for p in dev_ask]
        tail = len(prefix) + 1
    dev_talk = [encode(tok, system, q) for q in req["talk dev"]]
    dev_act = [] if ask == "act" else [encode(tok, system, q) for q in req.get("act dev", [])]
    dev = dev_ask + dev_talk + dev_act
    tails = [tail] * len(dev_ask) + [1] * (len(dev_talk) + len(dev_act))
    k, m = len(dev_ask), len(dev_talk)

    def score(out, base=None):
        ask_logp = [torch.log_softmax(lg, -1) for lg in out[:k]]
        row = {"start": sum(lp[0, start].exp().sum().item() for lp in ask_logp) / k,
               "first": [tok.decode([int(lp[0].argmax())]) for lp in ask_logp]}
        if prefix is not None:
            row["prefix_logp"] = sum(sum(lp[j, prefix[j]].item() for j in range(len(prefix)))
                                     for lp in ask_logp) / k
        talk = torch.log_softmax(torch.stack([lg[-1] for lg in out[k:k + m]]), -1)
        row["p$_talk"] = talk[:, dollar].exp().sum(-1).mean().item()
        if dev_act:
            act = torch.log_softmax(torch.stack([lg[-1] for lg in out[k + m:]]), -1)
            row["p$_act"] = act[:, dollar].exp().sum(-1).mean().item()
        if base is not None:
            base_talk = torch.log_softmax(torch.stack([lg[-1] for lg in base[k:k + m]]), -1)
            row["kl_talk"] = (base_talk.exp() * (base_talk - talk)).sum(-1).mean().item()
        return row

    cols = ["start"] + (["prefix_logp"] if prefix is not None else []) + (["p$_act"] if dev_act else []) \
        + ["p$_talk", "kl_talk"]
    print(f"{'candidate':>16} " + " ".join(f"{c:>11}" for c in cols) + f"   first token, '{ask}' dev", flush=True)

    def show(name, row):
        print(f"{name:>16} " + " ".join(f"{row.get(c, 0):11.3f}" for c in cols) + "   "
              + " ".join(repr(f) for f in row["first"][:6]), flush=True)

    base = s.run(dev, tails=tails)
    rows = {"baseline": score(base)}
    show("baseline", rows["baseline"])
    lo, hi = int(s.n_layers * args.from_depth), int(s.n_layers * args.to_depth)
    candidates = {}
    for layer in range(lo, hi + 1, args.step):
        for design, d in designs.items():
            r = d[layer] / d[layer].norm()
            name = f"{design}:L{layer}"
            rows[name] = score(s.run(dev, r=r, tails=tails), base)
            candidates[name] = r
            show(name, rows[name])

    keep = [c for c in candidates if rows[c]["kl_talk"] < args.max_kl
            and rows[c]["p$_talk"] < max(args.max_talk_dollar, rows["baseline"]["p$_talk"] + 0.05)
            and rows[c].get("p$_act", 1) >= rows["baseline"].get("p$_act", 1) - 0.1]
    if prefix is None:
        keep = [c for c in keep if rows[c]["start"] > rows["baseline"]["start"]]
        pick = max(keep, key=lambda c: rows[c]["start"]) if keep else None
    else:
        keep = [c for c in keep if rows[c]["prefix_logp"] < rows["baseline"]["prefix_logp"]]
        pick = min(keep, key=lambda c: rows[c]["prefix_logp"]) if keep else None
    print(f"pick: {pick} (best on the target with kl_talk < {args.max_kl}, words-only replies not pushed to $, "
          f"and computer questions still answered with $)")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"target": args.target, "candidates": candidates, "scores": rows, "pick": pick,
                "model": str(args.model), "requests": Path(args.requests).read_text(encoding="utf-8"),
                "system": system}, args.out)
    print(f"saved {args.out}")


def apply(args):
    found = torch.load(args.directions)
    name = args.candidate or found["pick"]
    r = found["candidates"][name].float()
    w = Weights(args.model)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for f in Path(args.model).iterdir():
        if f.is_file() and not f.name.endswith(".safetensors") and f.name != "model.safetensors.index.json":
            shutil.copy2(f, out / f.name)

    shards, group, size, weight_map, total = [], {}, 0, {}, 0

    def flush():
        nonlocal group, size
        if group:
            file = f"part-{len(shards):05d}.safetensors"
            save_file(group, out / file, metadata={"format": "pt"})
            shards.append((file, list(group)))
            group, size = {}, 0

    edited = 0
    for key in w.where:
        t = w.get(key)
        if key == EMBED:
            flush()
            new = torch.empty_like(t)
            for a in range(0, t.shape[0], 16384):
                new[a:a + 16384] = ablate(t[a:a + 16384].float(), r).to(t.dtype)
            t, edited = new, edited + 1
        elif key.endswith(WRITERS):
            tf = t.float()
            t = (tf - torch.outer(r, r @ tf) if tf.dim() == 2 else ablate(tf, r)).to(t.dtype)
            edited += 1
        group[key] = t.contiguous()
        size += t.numel() * t.element_size()
        total += t.numel() * t.element_size()
        if size > 1 << 30:
            flush()
    flush()
    for i, (file, keys) in enumerate(shards):
        final = f"model-{i + 1:05d}-of-{len(shards):05d}.safetensors"
        (out / file).rename(out / final)
        weight_map.update({k: final for k in keys})
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total},
                                                                  "weight_map": weight_map}, indent=2))
    (out / "ABLATION.json").write_text(json.dumps({"source": str(args.model), "target": found.get("target"),
                                                   "direction": name, "edited_tensors": edited,
                                                   "score": found["scores"][name]}, indent=2, default=str))
    print(f"{name}: orthogonalized {edited} tensors -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("find")
    f.add_argument("--model", required=True)
    f.add_argument("--target", choices=TARGETS, default="helpless")
    f.add_argument("--requests", required=True)
    f.add_argument("--system", required=True)
    f.add_argument("--out", required=True)
    f.add_argument("--from-depth", type=float, default=0.2)
    f.add_argument("--to-depth", type=float, default=0.8)
    f.add_argument("--step", type=int, default=2)
    f.add_argument("--max-kl", type=float, default=0.1)
    f.add_argument("--max-talk-dollar", type=float, default=0.1)
    f.add_argument("--threads", type=int, default=12)
    a = sub.add_parser("apply")
    a.add_argument("--model", required=True)
    a.add_argument("--directions", required=True)
    a.add_argument("--candidate")
    a.add_argument("--out", required=True)
    args = p.parse_args()
    find(args) if args.command == "find" else apply(args)


if __name__ == "__main__":
    main()
