"""Chat templates and the text tool-call protocol for models served as raw completions.

The checkpoint's ``chat_template`` is data from a downloaded model, so it is rendered in Jinja2's
immutable sandbox. Tool calls are requested as ``<tool_call>{json}</tool_call>`` blocks.
"""

from __future__ import annotations

import json
import re
import uuid
from dataclasses import dataclass, field
from pathlib import Path

TOOL_OPEN, TOOL_CLOSE = "<tool_call>", "</tool_call>"
_CALL_RE = re.compile(r"<tool_call>\s*(.*?)\s*(?:</tool_call>|\Z)", re.S)
_FENCE_RE = re.compile(r"^```(?:json)?\s*|\s*```$")


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict | None  # None when the JSON was malformed
    raw_arguments: str = ""
    error: str | None = None

    def to_message(self) -> dict:
        args = self.raw_arguments if self.arguments is None else json.dumps(self.arguments, ensure_ascii=False)
        return {"id": self.id, "type": "function", "function": {"name": self.name, "arguments": args}}


def new_call_id() -> str:
    return "call_" + uuid.uuid4().hex[:16]


def parse_arguments(raw) -> tuple[dict | None, str | None]:
    if isinstance(raw, dict):
        return raw, None
    if raw is None or (isinstance(raw, str) and not raw.strip()):
        return {}, None
    try:
        val = json.loads(raw)
    except (TypeError, json.JSONDecodeError) as e:
        return None, f"arguments are not valid JSON: {e}"
    if not isinstance(val, dict):
        return None, "arguments must be a JSON object"
    return val, None


def parse_text_tool_calls(text: str) -> tuple[str, list[ToolCall]]:
    """Split model text into prose and tool calls. Malformed calls are returned with ``error`` set
    so the loop can tell the model what went wrong."""
    calls: list[ToolCall] = []
    for m in _CALL_RE.finditer(text):
        body = _FENCE_RE.sub("", m.group(1).strip())
        if not body:
            continue
        try:
            obj = json.loads(body)
        except json.JSONDecodeError as e:
            calls.append(ToolCall(new_call_id(), "?", None, body, f"tool call is not valid JSON: {e}"))
            continue
        if not isinstance(obj, dict) or not isinstance(obj.get("name"), str):
            calls.append(ToolCall(new_call_id(), "?", None, body, 'tool call needs {"name": ..., "arguments": {...}}'))
            continue
        args, err = parse_arguments(obj.get("arguments", obj.get("parameters", {})))
        calls.append(ToolCall(new_call_id(), obj["name"], args, json.dumps(obj.get("arguments")), err))
    prose = _CALL_RE.sub("", text).strip()
    return prose, calls


def tools_prompt(tool_specs: list[dict]) -> str:
    lines = ["# Tools", "",
             "You can call tools. To call one, write exactly one line per call:",
             f'{TOOL_OPEN}{{"name": "<tool name>", "arguments": {{...}}}}{TOOL_CLOSE}',
             "Then stop and wait: the result comes back in the next message. Available tools:", ""]
    for spec in tool_specs:
        f = spec["function"]
        lines.append(f"## {f['name']}\n{f['description']}\nparameters: {json.dumps(f['parameters'])}\n")
    return "\n".join(lines)


def to_text_protocol(messages: list[dict]) -> list[dict]:
    """Rewrite native tool-calling messages for a model that only sees text."""
    out = []
    for m in messages:
        if m["role"] == "assistant" and m.get("tool_calls"):
            calls = "\n".join(
                f'{TOOL_OPEN}{{"name": {json.dumps(c["function"]["name"])}, "arguments": {c["function"]["arguments"] or "{}"}}}{TOOL_CLOSE}'
                for c in m["tool_calls"])
            content = ((m.get("content") or "") + "\n" + calls).strip()
            out.append({"role": "assistant", "content": content})
        elif m["role"] == "tool":
            out.append({"role": "tool", "content": m.get("content") or "", "name": m.get("name", "")})
        else:
            out.append({"role": m["role"], "content": m.get("content") or ""})
    return out


@dataclass
class ChatTemplate:
    template: str | None = None
    bos_token: str = ""
    eos_token: str = ""
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_checkpoint(cls, path: str | Path) -> "ChatTemplate":
        p = Path(path) / "tokenizer_config.json"
        if not p.is_file() or p.stat().st_size > 8 << 20:
            return cls()
        cfg = json.loads(p.read_text(encoding="utf-8"))

        def tok(v):
            return v.get("content", "") if isinstance(v, dict) else (v or "")

        tmpl = cfg.get("chat_template")
        if isinstance(tmpl, list):  # named templates
            tmpl = next((t.get("template") for t in tmpl if t.get("name") == "default"), None)
        return cls(tmpl if isinstance(tmpl, str) else None, tok(cfg.get("bos_token")), tok(cfg.get("eos_token")))

    def render(self, messages: list[dict]) -> str:
        if self.template:
            try:
                return self._render_jinja(messages)
            except Exception:
                # many templates reject the 'tool' role; retry with tool results as user turns
                try:
                    return self._render_jinja([
                        {"role": "user", "content": "[tool result]\n" + m["content"]} if m["role"] == "tool" else m
                        for m in messages])
                except Exception:
                    pass
        return self._render_plain(messages)

    def _render_jinja(self, messages):
        from jinja2.sandbox import ImmutableSandboxedEnvironment

        def raise_exception(msg):
            raise ValueError(msg)

        env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)
        env.globals["raise_exception"] = raise_exception
        return env.from_string(self.template).render(
            messages=messages, bos_token=self.bos_token, eos_token=self.eos_token,
            add_generation_prompt=True, **self.extra)

    def _render_plain(self, messages):
        parts = [self.bos_token]
        for m in messages:
            parts.append(f"<|{m['role']}|>\n{m.get('content') or ''}\n")
        parts.append("<|assistant|>\n")
        return "".join(parts)
