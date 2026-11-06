"""Model backends: any OpenAI-compatible HTTP endpoint, the engine in-process, and a scripted mock."""

from __future__ import annotations

import ipaddress
import json
import os
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from typing import Callable

from .chat_format import (TOOL_CLOSE, ChatTemplate, ToolCall, new_call_id, parse_arguments, parse_text_tool_calls,
                          to_text_protocol, tools_prompt)
from .config import BackendConfig


class BackendError(Exception):
    pass


@dataclass
class AssistantTurn:
    text: str = ""
    tool_calls: list[ToolCall] = field(default_factory=list)
    usage: dict | None = None
    finish_reason: str | None = None

    def to_message(self) -> dict:
        m = {"role": "assistant", "content": self.text}
        if self.tool_calls:
            m["tool_calls"] = [c.to_message() for c in self.tool_calls]
        return m


OnText = Callable[[str], None]


class Backend:
    name = "backend"
    tool_protocol = "native"

    def chat(self, messages: list[dict], tools: list[dict], on_text: OnText | None = None) -> AssistantTurn:
        raise NotImplementedError

    def count_tokens(self, text: str) -> int:
        return max(1, int(len(text) / 3.5))


def _is_loopback(host: str) -> bool:
    if host in ("localhost",):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def _text_protocol_messages(messages: list[dict], tools: list[dict], tool_role: str = "tool") -> list[dict]:
    msgs = to_text_protocol(messages)
    if tools:
        prompt = tools_prompt(tools)
        if msgs and msgs[0]["role"] == "system":
            msgs[0] = {"role": "system", "content": msgs[0]["content"] + "\n\n" + prompt}
        else:
            msgs.insert(0, {"role": "system", "content": prompt})
    if tool_role != "tool":
        msgs = [{"role": tool_role, "content": "[tool result]\n" + m["content"]} if m["role"] == "tool" else
                {"role": m["role"], "content": m["content"]} for m in msgs]
    return msgs


class OpenAIBackend(Backend):
    """``POST {endpoint}/chat/completions`` with server-sent-event streaming."""

    name = "openai"

    def __init__(self, cfg: BackendConfig, opener: urllib.request.OpenerDirector | None = None):
        u = urllib.parse.urlsplit(cfg.endpoint)
        if u.scheme not in ("http", "https") or not u.hostname:
            raise BackendError(f"endpoint must be an http(s) URL, got {cfg.endpoint!r}")
        if u.username or u.password:
            raise BackendError("do not put credentials in the endpoint URL; use api_key_env")
        if u.scheme == "http" and not _is_loopback(u.hostname) and not cfg.allow_insecure_http:
            raise BackendError("refusing plain http to a non-loopback host (the API key and your code would "
                               "travel unencrypted); use https or set backend.allow_insecure_http")
        self.cfg = cfg
        self.url = cfg.endpoint.rstrip("/") + "/chat/completions"
        self.tool_protocol = cfg.tool_protocol
        self.api_key = os.environ.get(cfg.api_key_env) if cfg.api_key_env else None
        if cfg.api_key_env and not self.api_key:
            raise BackendError(f"environment variable {cfg.api_key_env} is not set")
        self.opener = opener or urllib.request.build_opener()

    @property
    def remote(self) -> bool:
        return not _is_loopback(urllib.parse.urlsplit(self.url).hostname or "")

    def _payload(self, messages, tools) -> dict:
        p = {"model": self.cfg.model, "stream": True, "temperature": self.cfg.temperature,
             "max_tokens": self.cfg.max_tokens}
        if self.tool_protocol == "native":
            p["messages"] = messages
            if tools:
                p["tools"] = tools
                p["tool_choice"] = "auto"
        else:
            p["messages"] = _text_protocol_messages(messages, tools, tool_role="user")
            p["stop"] = [TOOL_CLOSE]
        return p

    def chat(self, messages, tools, on_text=None) -> AssistantTurn:
        body = json.dumps(self._payload(messages, tools)).encode()
        headers = {"Content-Type": "application/json", "Accept": "text/event-stream, application/json",
                   "User-Agent": "boilerplate-agent/0.1"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        req = urllib.request.Request(self.url, data=body, headers=headers, method="POST")
        try:
            resp = self.opener.open(req, timeout=self.cfg.timeout)
        except urllib.error.HTTPError as e:
            detail = e.read(2000).decode("utf-8", "replace")
            raise BackendError(f"HTTP {e.code} from model endpoint: {detail}") from None
        except (urllib.error.URLError, OSError) as e:
            raise BackendError(f"cannot reach model endpoint {self.url}: {e}") from None
        with resp:
            ctype = resp.headers.get("Content-Type", "")
            if "text/event-stream" in ctype:
                turn = self._read_stream(resp, on_text)
            else:
                data = resp.read(self.cfg.max_response_bytes + 1)
                if len(data) > self.cfg.max_response_bytes:
                    raise BackendError("model response too large")
                turn = self._from_json(json.loads(data), on_text)
        if self.tool_protocol == "text":
            text, calls = parse_text_tool_calls(turn.text + (TOOL_CLOSE if turn.text.count("<tool_call>") >
                                                               turn.text.count(TOOL_CLOSE) else ""))
            turn.text, turn.tool_calls = text, calls
        return turn

    def _from_json(self, data, on_text) -> AssistantTurn:
        try:
            choice = data["choices"][0]
            msg = choice["message"]
        except (KeyError, IndexError, TypeError):
            raise BackendError(f"unexpected response: {str(data)[:500]}") from None
        text = msg.get("content") or ""
        if on_text and text:
            on_text(text)
        calls = []
        for c in msg.get("tool_calls") or []:
            fn = c.get("function", {})
            args, err = parse_arguments(fn.get("arguments"))
            calls.append(ToolCall(c.get("id") or new_call_id(), fn.get("name", "?"), args,
                                  fn.get("arguments") or "", err))
        return AssistantTurn(text, calls, data.get("usage"), choice.get("finish_reason"))

    def _read_stream(self, resp, on_text) -> AssistantTurn:
        text: list[str] = []
        calls: dict[int, dict] = {}
        usage, finish, total = None, None, 0
        for raw in resp:
            total += len(raw)
            if total > self.cfg.max_response_bytes:
                raise BackendError("model response too large")
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            data = line[5:].strip()
            if data == "[DONE]":
                break
            try:
                ev = json.loads(data)
            except json.JSONDecodeError:
                continue
            if ev.get("error"):
                raise BackendError(f"model endpoint error: {str(ev['error'])[:500]}")
            usage = ev.get("usage") or usage
            for ch in ev.get("choices") or []:
                finish = ch.get("finish_reason") or finish
                delta = ch.get("delta") or {}
                if delta.get("content"):
                    text.append(delta["content"])
                    if on_text:
                        on_text(delta["content"])
                for tc in delta.get("tool_calls") or []:
                    slot = calls.setdefault(tc.get("index", 0), {"id": None, "name": "", "args": ""})
                    slot["id"] = tc.get("id") or slot["id"]
                    fn = tc.get("function") or {}
                    slot["name"] += fn.get("name") or ""
                    slot["args"] += fn.get("arguments") or ""
        out = []
        for _, s in sorted(calls.items()):
            args, err = parse_arguments(s["args"])
            out.append(ToolCall(s["id"] or new_call_id(), s["name"] or "?", args, s["args"], err))
        return AssistantTurn("".join(text), out, usage, finish)


class _Stop(Exception):
    pass


def generate_chat(engine, template: ChatTemplate, messages: list[dict], tools: list[dict], *,
                  max_tokens: int, temperature: float, max_seq_len: int, on_text: OnText | None = None,
                  seed: int | None = None) -> AssistantTurn:
    """Run one assistant turn on a Boilerplate ``Engine`` with the text tool-call protocol.
    Shared by the in-process backend and ``boilerplate serve``."""
    tok = engine.tokenizer
    if tok is None:
        raise BackendError("the checkpoint has no tokenizer.json")
    prompt = template.render(_text_protocol_messages(messages, tools))
    ids = tok.encode(prompt, add_special_tokens=False).ids if _accepts_special(tok) else tok.encode(prompt).ids
    budget = min(max_tokens, max_seq_len - len(ids))
    if budget <= 0:
        raise BackendError(f"prompt is {len(ids)} tokens, which fills the {max_seq_len}-token context; compact first")
    out_ids: list[int] = []
    emitted = [""]

    def stream(_i, t):
        out_ids.append(t)
        full = tok.decode(out_ids)
        if full.endswith("�"):
            return  # incomplete UTF-8 sequence; wait for more tokens
        new = full[len(emitted[0]):]
        emitted[0] = full
        if new and on_text:
            on_text(new)
        if TOOL_CLOSE in full[-len(new) - len(TOOL_CLOSE):]:
            raise _Stop

    finish = "stop"
    try:
        res = engine.generate([ids], max_new_tokens=budget, temperature=temperature, seed=seed, stream=stream)
        if not res.finished[0]:
            finish = "length"
    except _Stop:
        finish = "tool_calls"
    text = tok.decode(out_ids)
    prose, calls = parse_text_tool_calls(text)
    return AssistantTurn(prose, calls, {"prompt_tokens": len(ids), "completion_tokens": len(out_ids)},
                         "tool_calls" if calls else finish)


def _accepts_special(tok) -> bool:
    try:
        tok.encode("", add_special_tokens=False)
        return True
    except TypeError:
        return False


class LocalEngineBackend(Backend):
    """The Boilerplate engine in this process."""

    name = "local"
    tool_protocol = "text"

    def __init__(self, cfg: BackendConfig, engine=None, template: ChatTemplate | None = None):
        if engine is None:
            if not cfg.checkpoint:
                raise BackendError("local backend needs a checkpoint directory")
            from ..engine import Engine

            engine = Engine.load(cfg.checkpoint, max_seq_len=cfg.max_seq_len)
        self.engine = engine
        self.template = template or (ChatTemplate.from_checkpoint(cfg.checkpoint) if cfg.checkpoint else ChatTemplate())
        self.cfg = cfg
        self.remote = False

    def chat(self, messages, tools, on_text=None) -> AssistantTurn:
        return generate_chat(self.engine, self.template, messages, tools, max_tokens=self.cfg.max_tokens,
                             temperature=self.cfg.temperature, max_seq_len=self.cfg.max_seq_len, on_text=on_text)

    def count_tokens(self, text: str) -> int:
        tok = self.engine.tokenizer
        return len(tok.encode(text).ids) if tok is not None else super().count_tokens(text)


class MockBackend(Backend):
    """Scripted turns for tests. Each item is an ``AssistantTurn`` or a function of the messages."""

    name = "mock"

    def __init__(self, script: list, tool_protocol: str = "native"):
        self.script = list(script)
        self.calls: list[tuple[list[dict], list[dict]]] = []
        self.tool_protocol = tool_protocol
        self.remote = False

    def chat(self, messages, tools, on_text=None) -> AssistantTurn:
        self.calls.append((json.loads(json.dumps(messages)), tools))
        if not self.script:
            return AssistantTurn("(mock script exhausted)")
        item = self.script.pop(0)
        turn = item(messages) if callable(item) else item
        if isinstance(turn, str):
            prose, calls = parse_text_tool_calls(turn)
            turn = AssistantTurn(prose, calls)
        if on_text and turn.text:
            on_text(turn.text)
        return turn


def make_backend(cfg: BackendConfig) -> Backend:
    if cfg.kind == "local":
        return LocalEngineBackend(cfg)
    if cfg.kind == "openai":
        return OpenAIBackend(cfg)
    raise BackendError(f"backend kind {cfg.kind!r} cannot be built from config")
