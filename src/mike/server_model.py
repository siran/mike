"""A kernel served by llama.cpp's llama-server (OpenAI-compatible API); used to examine candidate models.

MIKE_TOOLS=native: instead of "$ " lines, the model gets one tool, `run`, in its own trained tool-calling format.
The session still sees "$ command" in its history; this class translates both ways.
"""
from __future__ import annotations

import json
import os
import time
import urllib.request

RUN_TOOL = {"type": "function", "function": {
    "name": "run",
    "description": "Run a shell command in your own Linux computer and see its output.",
    "parameters": {"type": "object", "required": ["command"], "properties": {
        "command": {"type": "string", "description": "the command, as you would type it in a terminal"}}}}}


class ServerModel:
    def __init__(self, url: str, max_new_tokens: int = 1024, timeout: float = 900):
        self.url = url.rstrip("/")
        self.max_new_tokens = max_new_tokens
        self.timeout = timeout
        self.thinking = os.environ.get("MIKE_THINKING", "off").lower() in ("on", "1", "true", "yes")
        self.native = os.environ.get("MIKE_TOOLS") == "native"
        self._calls: dict[str, tuple[str, str]] = {}  # kept reply text -> (what he said, the command he ran)
        props = self._call("/props")
        name = os.path.basename(str(props.get("model_path", ""))) or "llama-server"
        self.metadata = {"model": name, "server": self.url, "parameters": 0, "torch_threads": 0,
                         "decoding": "greedy (temperature 0)", "thinking": self.thinking,
                         "tools": "native (run)" if self.native else "$ lines"}

    def _call(self, path: str, body: dict | None = None) -> dict:
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self.url + path, data=data,
                                         headers={"Content-Type": "application/json"} if data else {})
        with urllib.request.urlopen(request, timeout=self.timeout) as response:
            return json.loads(response.read().decode())

    def _messages(self, messages: list[dict]) -> list[dict]:
        """The session's "$ command" / "output:" turns as native tool calls and tool results."""
        if not self.native:
            return messages
        out, n = [], 0
        for m in messages:
            if m["role"] == "assistant" and m["content"] in self._calls:
                said, command = self._calls[m["content"]]
                n += 1
                out.append({"role": "assistant", "content": said, "tool_calls": [{
                    "id": f"call_{n}", "type": "function",
                    "function": {"name": "run", "arguments": json.dumps({"command": command})}}]})
            elif m["role"] == "user" and m["content"].startswith("output:") and out and out[-1].get("tool_calls"):
                out.append({"role": "tool", "tool_call_id": out[-1]["tool_calls"][0]["id"],
                            "content": m["content"][len("output:"):].lstrip("\n")})
            else:
                out.append(m)
        return out

    def _request(self, messages: list[dict]) -> dict:
        body = {"messages": self._messages(messages), "chat_template_kwargs": {"enable_thinking": self.thinking}}
        if self.native:
            body["tools"] = [RUN_TOOL]
        return body

    def count_tokens(self, messages: list[dict]) -> int:
        prompt = self._call("/apply-template", self._request(messages))["prompt"]
        return len(self._call("/tokenize", {"content": prompt})["tokens"])

    def generate(self, messages: list[dict], stop=None, max_new_tokens: int | None = None) -> dict:
        body = self._request(messages)
        body.update({"max_tokens": max_new_tokens or self.max_new_tokens, "temperature": 0, "cache_prompt": True})
        start = time.perf_counter()
        result = self._call("/v1/chat/completions", body)
        elapsed = time.perf_counter() - start
        message = result["choices"][0]["message"]
        raw = (message.get("content") or "").strip()
        usage = result.get("usage", {})
        generation = {"input_tokens": usage.get("prompt_tokens", 0),
                      "output_tokens": usage.get("completion_tokens", 0), "inference_seconds": elapsed,
                      "reused_tokens": (result.get("timings") or {}).get("cache_n", 0)}
        if self.native:
            calls = message.get("tool_calls") or []
            command = None
            if calls:
                try:
                    command = json.loads(calls[0]["function"]["arguments"]).get("command")
                except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
                    generation["bad_tool_call"] = calls[0]
                generation["tool_calls"] = calls
            kept = raw if command is None else (raw + "\n" if raw else "") + f"$ {command}"
            if command is not None:
                self._calls[kept] = (raw, command)
            generation["raw"] = kept
            generation["parsed"] = (raw.splitlines(), command, kept)
        else:
            if stop is not None:  # greedy: the text up to the first stop point is what an early stop would give
                for i, ch in enumerate(raw):
                    if ch == "\n" and stop(raw[:i + 1]):
                        raw = raw[:i + 1].strip()
                        break
            generation["raw"] = raw
        reasoning = (message.get("reasoning_content") or "").strip()
        if reasoning:
            generation["reasoning"] = reasoning
            print("  (thinking) " + reasoning.replace("\n", "\n  (thinking) "), flush=True)
        return generation
