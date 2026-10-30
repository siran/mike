"""Remove one direction from a model's residual stream, in its weights (directional ablation, Arditi et al. 2024).

  find   activations for contrasting prompts -> candidate directions, each scored on held-out prompts
  apply  orthogonalize every residual-stream writer against the chosen direction; save a new model

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

    @torch.inference_mode()
    def run(self, prompts, r=None, collect=False):
        """Last-position logits for each prompt; with collect, the residual stream entering each layer there too."""
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
        x = self.norm(torch.stack([h[-1] for h in hs]))
        logits = torch.cat([x @ self.unembed[a:a + 32768].float().T
                            for a in range(0, self.unembed.shape[0], 32768)], dim=1)
        if r is not None and self.tied:  # the edited embedding is also the unembedding
            u_r = torch.cat([self.unembed[a:a + 32768].float() @ r for a in range(0, self.unembed.shape[0], 32768)])
            logits = logits - (x @ r).unsqueeze(-1) * u_r
        return (logits, torch.stack(acts)) if collect else logits


def encode(tokenizer, system, user):
    ids = tokenizer.apply_chat_template([{"role": "system", "content": system}, {"role": "user", "content": user}],
                                        add_generation_prompt=True, tokenize=True)
    return torch.tensor(ids if isinstance(ids, list) else ids["input_ids"])


def find(args):
    torch.set_num_threads(args.threads)
    req = read_requests(args.requests)
    system = Path(args.system).read_text(encoding="utf-8").strip()
    helpless = " ".join(req["helpless system"])
    tok = AutoTokenizer.from_pretrained(args.model)
    s = Streamer(args.model)
    dollar = torch.tensor([i for i in range(len(tok)) if tok.decode([i]).lstrip().startswith("$")])
    print(f"{s.config.architectures[0]}, {s.n_layers} layers, {len(dollar)} tokens that start with $", flush=True)

    t = time.time()
    a_help = [encode(tok, helpless, q) for q in req["act train"]]
    a_mike = [encode(tok, system, q) for q in req["act train"]]
    t_mike = [encode(tok, system, q) for q in req["talk train"]]
    train_logits, acts = s.run(a_help + a_mike + t_mike, collect=True)
    n = len(a_help)
    help_m, mike_m, talk_m = acts[:, :n].mean(1), acts[:, n:2 * n].mean(1), acts[:, 2 * n:].mean(1)
    designs = {"situation": help_m - mike_m, "request": mike_m - talk_m}
    acted = torch.log_softmax(train_logits[n:2 * n], -1)[:, dollar].exp().sum(-1) > 0.5
    print(f"act train, Mike's prompt: {int(acted.sum())}/{n} start with $", flush=True)
    if 3 <= int(acted.sum()) <= n - 3:  # his own replies: the ones he didn't act on minus the ones he did
        designs["reflex"] = acts[:, n:2 * n][:, ~acted].mean(1) - acts[:, n:2 * n][:, acted].mean(1)
    print(f"activations collected in {time.time() - t:.0f} s", flush=True)

    dev_act = [encode(tok, system, q) for q in req["act dev"]]
    dev_talk = [encode(tok, system, q) for q in req["talk dev"]]
    dev = dev_act + dev_talk
    k = len(dev_act)

    def score(logits, base=None):
        logp = torch.log_softmax(logits, -1)
        p_dollar = logp[:, dollar].exp().sum(-1)
        first = [tok.decode([i]) for i in logits.argmax(-1)]
        row = {"p$_act": p_dollar[:k].mean().item(), "p$_talk": p_dollar[k:].mean().item(),
               "$first_act": sum(f.lstrip().startswith("$") for f in first[:k]), "first_act": first[:k]}
        if base is not None:
            base_logp = torch.log_softmax(base, -1)
            row["kl_talk"] = (base_logp[k:].exp() * (base_logp[k:] - logp[k:])).sum(-1).mean().item()
        return row

    base = s.run(dev)
    rows = {"baseline": score(base)}
    print(f"{'candidate':>16} {'p$_act':>7} {'p$_talk':>7} {'kl_talk':>7} {'$first':>6}  first tokens (act dev)", flush=True)

    def show(name, row):
        print(f"{name:>16} {row['p$_act']:7.3f} {row['p$_talk']:7.3f} {row.get('kl_talk', 0):7.3f} "
              f"{row['$first_act']:>3}/{k:<2}  {' '.join(repr(f) for f in row['first_act'][:6])}", flush=True)

    show("baseline", rows["baseline"])
    lo, hi = int(s.n_layers * args.from_depth), int(s.n_layers * args.to_depth)
    candidates = {}
    for layer in range(lo, hi + 1, args.step):
        for design, d in designs.items():
            r = d[layer] / d[layer].norm()
            name = f"{design}:L{layer}"
            rows[name] = score(s.run(dev, r=r), base)
            candidates[name] = r
            show(name, rows[name])

    ok = [n for n in candidates if rows[n]["kl_talk"] < args.max_kl and rows[n]["p$_talk"] < args.max_talk_dollar]
    pick = max(ok, key=lambda n: rows[n]["p$_act"]) if ok else None
    print(f"pick: {pick} (highest p$_act with kl_talk < {args.max_kl} and p$_talk < {args.max_talk_dollar})")
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"candidates": candidates, "scores": rows, "pick": pick, "model": str(args.model),
                "requests": Path(args.requests).read_text(encoding="utf-8"), "system": system}, args.out)
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
    (out / "ABLATION.json").write_text(json.dumps({"source": str(args.model), "direction": name,
                                                   "edited_tensors": edited, "score": found["scores"][name]},
                                                  indent=2, default=str))
    print(f"{name}: orthogonalized {edited} tensors -> {out}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="command", required=True)
    f = sub.add_parser("find")
    f.add_argument("--model", required=True)
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
