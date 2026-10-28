"""A kernel served by llama.cpp's llama-server (OpenAI-compatible API); used to examine candidate models."""
from __future__ import annotations

import json
import os
import time
import urllib.request


class ServerModel:
    def __init__(self, url: str, max_new_tokens: int = 1024, timeout: float = 900):
        self.url = url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        thinking = os.environ.get("MIKE_THINKING", "off").lower()
        self.thinking = thinking in ("on", "1", "true", "yes")
        props = self._call("/props")
        name = os.path.basename(str(props.get("model_path", ""))) or "llama-server"
        self.metadata = {"model": name, "server": self.url, "parameters": 0, "torch_threads": 0,
                         "decoding": "greedy (temperature 0)", "thinking": self.thinking}

    def _call(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data,
                                         headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    def count_tokens(self, messages: list[dict]) -> int:
        prompt = self._call("/apply-template", {"messages": messages})["prompt"]
        return len(self._call("/tokenize", {"content": prompt})["tokens"])

    def generate(self, messages: list[dict], stop=None, max_new_tokens: int | None = None) -> dict:
        body = {"messages": messages, "max_tokens": max_new_tokens or self.max_new_tokens,
                "temperature": 0, "cache_prompt": True,
                "chat_template_kwargs": {"enable_thinking": self.thinking}}
        start = time.perf_counter()
        result = self._call("/v1/chat/completions", body)
        elapsed = time.perf_counter() - start
        message = result["choices"][0]["message"]
        raw = (message.get("content") or "").strip()
        if stop is not None:  # greedy: the text up to the first stop point is what an early stop would give
            for i, ch in enumerate(raw):
                if ch == "\n" and stop(raw[:i + 1]):
                    raw = raw[:i + 1].strip()
                    break
        usage = result.get("usage", {})
        generation = {"raw": raw, "input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0), "inference_seconds": elapsed,
                      "reused_tokens": (result.get("timings") or {}).get("cache_n", 0)}
        reasoning = (message.get("reasoning_content") or "").strip()
        if reasoning:
            generation["reasoning"] = reasoning
            print("  (thinking) " + reasoning.replace("\n", "\n  (thinking) "), flush=True)
        return generation
