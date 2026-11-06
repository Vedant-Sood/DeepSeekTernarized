"""Minimal OpenAI-compatible HTTP server over the Boilerplate engine (development use).

    boilerplate serve ./checkpoint --port 8000
    curl http://127.0.0.1:8000/v1/chat/completions -d '{"messages":[{"role":"user","content":"hi"}]}'

Tools are supported through the text tool-call protocol (see boilerplate/agent/chat_format.py).
Generation is serialised: one request runs at a time. It binds to loopback unless an API key is
configured, compares keys in constant time, and caps request size and generation length.
"""

from __future__ import annotations

import hmac
import ipaddress
import json
import os
import threading
import time
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .agent.backends import BackendError, generate_chat
from .agent.chat_format import ChatTemplate

MAX_BODY = 8 << 20
MAX_MESSAGES = 2000


class ServeState:
    def __init__(self, engine, template: ChatTemplate, model_name: str, *, max_seq_len: int, max_tokens_cap: int,
                 api_key: str | None):
        self.engine, self.template, self.model_name = engine, template, model_name
        self.max_seq_len, self.max_tokens_cap, self.api_key = max_seq_len, max_tokens_cap, api_key
        self.lock = threading.Lock()


def _validate(body: dict, state: ServeState) -> tuple[list[dict], list[dict], int, float, bool]:
    msgs = body.get("messages")
    if not isinstance(msgs, list) or not msgs or len(msgs) > MAX_MESSAGES:
        raise ValueError(f"messages must be a non-empty list of at most {MAX_MESSAGES}")
    clean = []
    for m in msgs:
        if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool"):
            raise ValueError("each message needs a role of system, user, assistant or tool")
        content = m.get("content") or ""
        if isinstance(content, list):  # content parts: keep the text ones
            content = "".join(p.get("text", "") for p in content if isinstance(p, dict) and p.get("type") == "text")
        if not isinstance(content, str):
            raise ValueError("message content must be a string")
        out = {"role": m["role"], "content": content}
        if m["role"] == "assistant" and isinstance(m.get("tool_calls"), list):
            out["tool_calls"] = [c for c in m["tool_calls"] if isinstance(c, dict) and isinstance(c.get("function"), dict)]
        if m["role"] == "tool":
            out["name"] = str(m.get("name", ""))[:128]
        clean.append(out)
    tools = body.get("tools") or []
    if not isinstance(tools, list) or len(tools) > 128:
        raise ValueError("tools must be a list of at most 128")
    max_tokens = body.get("max_tokens", body.get("max_completion_tokens", 512))
    if not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1:
        raise ValueError("max_tokens must be a positive integer")
    temperature = body.get("temperature", 0.0)
    if not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
        raise ValueError("temperature must be between 0 and 2")
    return clean, tools, min(max_tokens, state.max_tokens_cap), float(temperature), bool(body.get("stream"))


def make_handler(state: ServeState):
    class Handler(BaseHTTPRequestHandler):
        server_version = "boilerplate-serve/0.1"
        sys_version = ""

        def log_message(self, fmt, *args):  # no request bodies or headers in logs
            print(f"[serve] {self.address_string()} {fmt % args}")

        def _json(self, code: int, obj: dict):
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _error(self, code: int, msg: str, kind: str = "invalid_request_error"):
            self._json(code, {"error": {"message": msg, "type": kind}})

        def _authorised(self) -> bool:
            if not state.api_key:
                return True
            got = self.headers.get("Authorization", "")
            want = f"Bearer {state.api_key}"
            return hmac.compare_digest(got.encode(), want.encode())

        def do_GET(self):
            if not self._authorised():
                return self._error(401, "missing or wrong API key", "authentication_error")
            if self.path.rstrip("/") in ("/v1/models", "/models"):
                return self._json(200, {"object": "list", "data": [
                    {"id": state.model_name, "object": "model", "owned_by": "boilerplate"}]})
            if self.path in ("/health", "/v1/health"):
                return self._json(200, {"status": "ok"})
            self._error(404, "not found")

        def do_POST(self):
            if not self._authorised():
                return self._error(401, "missing or wrong API key", "authentication_error")
            if self.path.rstrip("/") not in ("/v1/chat/completions", "/chat/completions"):
                return self._error(404, "not found")
            try:
                length = int(self.headers.get("Content-Length", "0"))
            except ValueError:
                return self._error(400, "bad Content-Length")
            if length <= 0 or length > MAX_BODY:
                return self._error(413, f"request body must be 1..{MAX_BODY} bytes")
            try:
                body = json.loads(self.rfile.read(length))
                if not isinstance(body, dict):
                    raise ValueError("body must be a JSON object")
                msgs, tools, max_tokens, temperature, stream = _validate(body, state)
            except (ValueError, json.JSONDecodeError) as e:
                return self._error(400, str(e))
            rid = "chatcmpl-" + uuid.uuid4().hex[:24]
            created = int(time.time())
            if stream:
                self.send_response(200)
                self.send_header("Content-Type", "text/event-stream")
                self.send_header("Cache-Control", "no-cache")
                self.end_headers()

            def chunk(delta: dict, finish=None):
                ev = {"id": rid, "object": "chat.completion.chunk", "created": created, "model": state.model_name,
                      "choices": [{"index": 0, "delta": delta, "finish_reason": finish}]}
                self.wfile.write(f"data: {json.dumps(ev)}\n\n".encode())
                self.wfile.flush()

            # In streaming mode text is sent as it is produced; tool-call markup is held back
            # (it is returned as structured tool_calls at the end).
            held = {"buf": ""}

            def on_text(t: str):
                if not stream:
                    return
                held["buf"] += t
                buf = held["buf"]
                cut = buf.find("<tool_call>")
                if cut < 0:
                    safe = len(buf)
                    for k in range(1, len("<tool_call>")):  # keep a possible partial tag
                        if buf.endswith("<tool_call>"[:k]):
                            safe = len(buf) - k
                            break
                    if safe:
                        chunk({"content": buf[:safe]})
                        held["buf"] = buf[safe:]
                elif cut > 0:
                    chunk({"content": buf[:cut]})
                    held["buf"] = buf[cut:]

            with state.lock:
                try:
                    turn = generate_chat(state.engine, state.template, msgs, tools, max_tokens=max_tokens,
                                         temperature=temperature, max_seq_len=state.max_seq_len, on_text=on_text)
                except BackendError as e:
                    if stream:
                        self.wfile.write(f"data: {json.dumps({'error': {'message': str(e)}})}\n\n".encode())
                        return
                    return self._error(400, str(e))
            calls = [c.to_message() for c in turn.tool_calls if c.error is None]
            finish = "tool_calls" if calls else (turn.finish_reason or "stop")
            usage = {**(turn.usage or {}), "total_tokens": sum((turn.usage or {}).values())}
            if stream:
                if held["buf"] and "<tool_call>" not in held["buf"]:
                    chunk({"content": held["buf"]})
                for i, c in enumerate(calls):
                    chunk({"tool_calls": [{"index": i, **c}]})
                chunk({}, finish)
                self.wfile.write(b"data: [DONE]\n\n")
                self.wfile.flush()
                return
            msg = {"role": "assistant", "content": turn.text or None}
            if calls:
                msg["tool_calls"] = calls
            self._json(200, {"id": rid, "object": "chat.completion", "created": created, "model": state.model_name,
                             "choices": [{"index": 0, "message": msg, "finish_reason": finish}], "usage": usage})

    return Handler


def _is_loopback(host: str) -> bool:
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def serve(checkpoint: str, host: str = "127.0.0.1", port: int = 8000, *, api_key_env: str | None = None,
          max_seq_len: int = 8192, max_tokens_cap: int = 4096, model_name: str | None = None,
          engine=None, template: ChatTemplate | None = None, ready=None) -> None:
    api_key = os.environ.get(api_key_env) if api_key_env else None
    if api_key_env and not api_key:
        raise SystemExit(f"environment variable {api_key_env} is not set")
    if not _is_loopback(host) and not api_key:
        raise SystemExit("refusing to listen on a non-loopback address without --api-key-env")
    if engine is None:
        from .engine import Engine

        engine = Engine.load(checkpoint, max_seq_len=max_seq_len)
    template = template or ChatTemplate.from_checkpoint(checkpoint)
    state = ServeState(engine, template, model_name or os.path.basename(os.path.normpath(checkpoint)),
                       max_seq_len=max_seq_len, max_tokens_cap=max_tokens_cap, api_key=api_key)
    httpd = ThreadingHTTPServer((host, port), make_handler(state))
    httpd.daemon_threads = True
    print(f"serving {state.model_name} on http://{host}:{httpd.server_address[1]}/v1")
    if ready is not None:
        ready(httpd)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        httpd.server_close()
