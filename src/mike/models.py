"""Frozen Hugging Face causal language models; explicit deterministic decoding."""
from __future__ import annotations

import json
import time

SYSTEM = """You are an agent answering questions about a database. Decide one action at a time. Reply with exactly one JSON object, no other text.
Search for facts about one entity: {"action":"search","entity":"ENTITY"}
Finish: {"action":"answer","value":"VALUE","evidence":["RECORD_ID"]}
When you cannot determine an answer, use null as value. Each search returns facts whose subject is the requested entity. Use returned objects to search further. Cite records you used. Do not guess unknown facts."""


class TransformersModel:
    def __init__(self, model_id: str, revision: str | None = None, max_new_tokens: int = 96,
                 threads: int | None = None, reuse_cache: bool = False, repetition_penalty: float = 1.0,
                 no_repeat_ngram_size: int = 0):
        import torch
        import transformers
        from transformers import AutoModelForCausalLM, AutoTokenizer
        if threads is not None:
            torch.set_num_threads(threads)
        self.torch = torch
        self.model_id = model_id
        self.tokenizer = AutoTokenizer.from_pretrained(model_id, revision=revision)
        self.model = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, torch_dtype=torch.float32)
        self.model.eval()
        self.max_new_tokens = max_new_tokens
        self.metadata = {"model": model_id, "requested_revision": revision,
                         "resolved_revision": getattr(self.model.config, "_commit_hash", None),
                         "parameters": sum(p.numel() for p in self.model.parameters()),
                         "torch": torch.__version__, "transformers": transformers.__version__,
                         "dtype": "float32", "device": "cpu", "decoding": "greedy",
                         "torch_threads": torch.get_num_threads(), "reuse_cache": reuse_cache,
                         "repetition_penalty": repetition_penalty, "no_repeat_ngram_size": no_repeat_ngram_size}
        self.decoding = {}
        if repetition_penalty != 1.0:
            self.decoding["repetition_penalty"] = repetition_penalty
        if no_repeat_ngram_size:
            self.decoding["no_repeat_ngram_size"] = no_repeat_ngram_size
        self.reuse_cache = reuse_cache
        self._cache, self._cached_ids = None, []

    def prompt(self, messages: list[dict]) -> str:
        return self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

    def count_tokens(self, messages: list[dict]) -> int:
        return len(self.tokenizer(self.prompt(messages))["input_ids"])

    def generate(self, messages: list[dict], stop=None, max_new_tokens: int | None = None) -> dict:
        """Greedy generation. `stop(text)` may end decoding early; it sees only generated text."""
        from transformers import StoppingCriteria, StoppingCriteriaList
        inputs = self.tokenizer(self.prompt(messages), return_tensors="pt")
        input_count = int(inputs["input_ids"].shape[-1])
        kwargs = {}
        if stop is not None:
            tokenizer, torch = self.tokenizer, self.torch

            class TextStop(StoppingCriteria):
                def __call__(self, input_ids, scores, **_):
                    text = tokenizer.decode(input_ids[0, input_count:], skip_special_tokens=True)
                    return torch.full((input_ids.shape[0],), bool(stop(text)), dtype=torch.bool)
            kwargs["stopping_criteria"] = StoppingCriteriaList([TextStop()])
        reused = 0
        if self.reuse_cache:
            reused, kwargs["past_key_values"] = self._reusable(inputs["input_ids"][0].tolist())
        start = time.perf_counter()
        with self.torch.inference_mode():
            result = self.model.generate(**inputs, max_new_tokens=max_new_tokens or self.max_new_tokens,
                                         do_sample=False, pad_token_id=self.tokenizer.eos_token_id,
                                         **self.decoding, **kwargs)
        elapsed = time.perf_counter() - start
        if self.reuse_cache:
            self._cached_ids = result[0][:self._cache.get_seq_length()].tolist()
        output = result[0][input_count:]
        raw = self.tokenizer.decode(output, skip_special_tokens=True).strip()
        return {"raw": raw, "input_tokens": input_count, "output_tokens": int(output.shape[-1]),
                "inference_seconds": elapsed, "reused_tokens": reused}

    def _reusable(self, ids: list[int]):
        """The key/value cache cut back to the prefix this prompt shares with the last sequence."""
        from transformers import DynamicCache
        common = 0
        for old, new in zip(self._cached_ids, ids):
            if old != new:
                break
            common += 1
        common = min(common, len(ids) - 1)
        if self._cache is None or common == 0:
            self._cache, common = DynamicCache(), 0
        else:
            self._cache.crop(common)
        self._cached_ids = []  # valid again only once this generation completes
        return common, self._cache

    def decide(self, state: dict) -> dict:
        """MIK-0 JSON interface: the whole state as one JSON user message."""
        return self.generate([{"role": "system", "content": SYSTEM},
                              {"role": "user", "content": json.dumps(state, ensure_ascii=False)}])


class ScriptedModel:
    """Test fixture, not an experimental model. Dict actions are JSON-encoded; strings are raw text."""
    def __init__(self, actions):
        self.actions = iter(actions)
        self.metadata = {"model": "scripted-test-fixture"}
        self.seen = []

    def _next(self):
        action = next(self.actions)
        raw = action if isinstance(action, str) else json.dumps(action)
        return {"raw": raw, "input_tokens": 0, "output_tokens": 0, "inference_seconds": 0.0}

    def decide(self, state):
        return self._next()

    def count_tokens(self, messages):
        return sum(len(m["content"]) for m in messages) // 4

    def generate(self, messages, stop=None, max_new_tokens=None):
        self.seen.append([dict(m) for m in messages])
        return self._next()
