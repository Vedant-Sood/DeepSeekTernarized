"""OpenAI-compatible HTTP server for Boilerplate checkpoints (``boilerplate serve``).

    boilerplate serve out/mode_d --port 8000
    curl localhost:8000/v1/completions -d '{"prompt": "Hello", "max_tokens": 16}'

Endpoints: ``GET /v1/models``, ``POST /v1/completions``, ``POST /v1/chat/completions`` (both with
``"stream": true`` as server-sent events, ending in ``data: [DONE]``), ``GET /health`` and
``GET /stats`` (expert cache and Mode B cache counters). Supported request fields: ``max_tokens``,
``temperature``, ``seed``, ``stop`` (string or list), ``n`` (completions batched in one engine
call), ``echo`` (completions). Unsupported sampling knobs (``top_p``, penalties, ...) are accepted
and ignored when they hold their neutral value and rejected otherwise, so a client never silently
gets a different sampler than it asked for.

The engine is not thread safe, so requests run one at a time on a worker thread behind a lock;
batching across requests is future work (the engines already batch the ``n`` choices of one
request). Chat prompts use the ``chat_template`` from ``tokenizer_config.json`` when there is one
(rendered with jinja2) and a plain ``role: content`` transcript otherwise.

The request logic lives in ``Server`` (no web framework involved, which is what most tests use);
``app()`` wraps it in Starlette and ``main`` runs it with uvicorn. Both are optional dependencies
(``pip install boilerplate[serve]``).

Security: the server binds to 127.0.0.1 by default. With ``--api-key`` (or ``BOILERPLATE_API_KEY``)
every endpoint except ``/health`` needs ``Authorization: Bearer <key>``; binding to a non-loopback
address without a key is refused unless ``--insecure-no-auth`` is given. Request bodies, prompt
length and the number of waiting requests are bounded (HTTP 413 / 400 / 503). A checkpoint's chat
template is rendered in Jinja's immutable sandbox, because ``tokenizer_config.json`` ships with the
(possibly untrusted) checkpoint. This is still a single-tenant research server: put a TLS-terminating
reverse proxy with its own rate limits in front of it before exposing it beyond a trusted network.
"""

from __future__ import annotations

import asyncio
import hmac
import ipaddress
import json
import os
import threading
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator


class RequestError(ValueError):
    """A client error, returned as HTTP 400 with an OpenAI-style error body."""


class _Stop(Exception):
    """Raised from the stream callback to end generation early (stop string or client gone)."""


MAX_BODY_BYTES = 4 << 20       # request body cap (HTTP 413 above it)
MAX_PROMPT_CHARS = 1 << 20      # prompt / rendered chat text cap, checked before tokenization
MAX_MESSAGES = 1024
MAX_STOP_CHARS = 256
MAX_PENDING = 16                # requests queued or running; more get HTTP 503

NEUTRAL = {"top_p": 1, "frequency_penalty": 0, "presence_penalty": 0, "logprobs": None, "best_of": 1,
           "logit_bias": None, "user": None, "suffix": None, "stream_options": None}


@dataclass
class Choice:
    text: str
    finish_reason: str  # "stop" (EOS or stop string) or "length"
    tokens: int        # generated tokens before EOS or the stop string


class _Decoder:
    """Incremental detokenizer: decodes the whole generated id list and emits the new suffix,
    holding back text that ends in an incomplete UTF-8 sequence or could still become a stop string."""

    def __init__(self, tokenizer, stop: list[str]):
        self.tok, self.stop = tokenizer, stop
        self.ids: list[int] = []
        self.sent = 0  # characters already emitted
        self.text = ""
        self.stopped = False

    def push(self, token: int) -> str:
        self.ids.append(token)
        self.text = self.tok.decode(self.ids)
        for s in self.stop:
            at = self.text.find(s)
            if at != -1:
                self.text = self.text[:at]
                self.stopped = True
                return self._emit(len(self.text))
        end = len(self.text)
        if self.text.endswith("\ufffd"):  # partial multi-byte character
            end = len(self.text.rstrip("\ufffd"))
        hold = max((len(s) - 1 for s in self.stop), default=0)
        for k in range(min(hold, end - self.sent), 0, -1):  # longest tail that starts a stop string
            if any(s.startswith(self.text[end - k:end]) for s in self.stop):
                end -= k
                break
        return self._emit(end)

    def flush(self) -> str:
        return self._emit(len(self.text))

    def _emit(self, end: int) -> str:
        if end <= self.sent:
            return ""
        out, self.sent = self.text[self.sent:end], end
        return out


class Server:
    def __init__(self, engine, model_name: str, chat_template: str | None = None,
                 default_max_tokens: int = 256, max_prompt_chars: int = MAX_PROMPT_CHARS):
        if engine.tokenizer is None:
            raise ValueError("serving needs tokenizer.json next to the checkpoint")
        self.engine, self.model_name = engine, model_name
        self.chat_template, self.default_max_tokens = chat_template, default_max_tokens
        self.max_prompt_chars = max_prompt_chars
        self._template = None
        self.lock = threading.Lock()
        self.started = int(time.time())
        self.requests = 0
        self.generated_tokens = 0

    @classmethod
    def load(cls, checkpoint: str | Path, name: str | None = None, **engine_args) -> "Server":
        from .engine import Engine

        checkpoint = Path(checkpoint)
        template = None
        cfg = checkpoint / "tokenizer_config.json"
        if cfg.exists():
            template = json.loads(cfg.read_text()).get("chat_template")
            if template is not None and not isinstance(template, str):
                template = None  # list-of-templates form: fall back to the plain transcript
        return cls(Engine.load(checkpoint, **engine_args), name or checkpoint.name, chat_template=template)

    # ------------------------------------------------------------------ request parsing
    def _params(self, body: dict) -> dict:
        for key, neutral in NEUTRAL.items():
            if key in body and body[key] != neutral and key not in ("user", "stream_options"):
                raise RequestError(f"`{key}` is not supported by this server (only {neutral!r})")
        stop = body.get("stop") or []
        if isinstance(stop, str):
            stop = [stop]
        if not isinstance(stop, list) or len(stop) > 4 or \
                not all(isinstance(s, str) and 0 < len(s) <= MAX_STOP_CHARS for s in stop):
            raise RequestError(f"`stop` must be a non-empty string or a list of up to 4 of them "
                               f"(at most {MAX_STOP_CHARS} characters each)")
        max_tokens = body.get("max_tokens", body.get("max_completion_tokens", self.default_max_tokens))
        n = body.get("n", 1)
        temperature = body.get("temperature", 1.0)
        if not _is_int(max_tokens) or max_tokens < 0:
            raise RequestError("`max_tokens` must be a nonnegative integer")
        if not _is_int(n) or not 1 <= n <= 8:
            raise RequestError("`n` must be between 1 and 8")
        if isinstance(temperature, bool) or not isinstance(temperature, (int, float)) or not 0 <= temperature <= 2:
            raise RequestError("`temperature` must be in [0, 2]")
        if n > 1 and temperature == 0:
            raise RequestError("`n` > 1 needs temperature > 0 (greedy choices would be identical)")
        seed = body.get("seed")
        if seed is not None and (not _is_int(seed) or not 0 <= seed < 2**63):
            raise RequestError("`seed` must be a nonnegative 64-bit integer")
        return {"stop": stop, "max_tokens": max_tokens, "n": n, "temperature": float(temperature), "seed": seed}

    def _check_text(self, text: str) -> str:
        if len(text) > self.max_prompt_chars:
            raise RequestError(f"the prompt is longer than {self.max_prompt_chars} characters")
        return text

    def encode(self, text: str) -> list[int]:
        return self.engine.tokenizer.encode(text).ids

    def chat_prompt(self, messages: list[dict]) -> str:
        if not isinstance(messages, list) or not messages:
            raise RequestError("`messages` must be a non-empty list")
        if len(messages) > MAX_MESSAGES:
            raise RequestError(f"at most {MAX_MESSAGES} messages are supported")
        for m in messages:
            if not isinstance(m, dict) or m.get("role") not in ("system", "user", "assistant", "tool") \
                    or not isinstance(m.get("content"), str):
                raise RequestError("each message needs a role and string content")
        self._check_text("".join(m["content"] for m in messages))
        # Only role and content reach the template: other client keys are not template inputs.
        messages = [{"role": m["role"], "content": m["content"]} for m in messages]
        if self.chat_template:
            import jinja2
            from jinja2.sandbox import ImmutableSandboxedEnvironment

            if self._template is None:
                # The template comes from the checkpoint's tokenizer_config.json, which may be
                # untrusted: a plain Environment would let it reach Python internals (SSTI -> RCE).
                env = ImmutableSandboxedEnvironment(trim_blocks=True, lstrip_blocks=True)

                def raise_exception(msg):
                    raise RequestError(str(msg))

                env.globals["raise_exception"] = raise_exception
                self._template = env.from_string(self.chat_template)
            try:
                text = self._template.render(messages=messages, add_generation_prompt=True,
                                             bos_token="", eos_token="")
            except RequestError:
                raise
            except jinja2.exceptions.SecurityError as e:
                raise RequestError("the checkpoint's chat template tried an unsafe operation") from e
            except jinja2.exceptions.TemplateError as e:
                raise RequestError(f"chat template error: {e}") from e
            return self._check_text(text)
        return self._check_text("".join(f"{m['role']}: {m['content']}\n" for m in messages) + "assistant:")

    # ------------------------------------------------------------------ generation
    def run(self, prompt_ids: list[int], p: dict, on_text: Callable[[int, str], None] | None = None,
            cancelled: Callable[[], bool] = lambda: False) -> list[Choice]:
        """Generate ``p['n']`` choices for one prompt; ``on_text(i, delta)`` streams text deltas."""
        if not prompt_ids:
            raise RequestError("the prompt is empty after tokenization")
        limit = _max_seq_len(self.engine)
        if limit is not None and len(prompt_ids) + p["max_tokens"] > limit:
            raise RequestError(f"prompt ({len(prompt_ids)} tokens) plus max_tokens ({p['max_tokens']}) "
                               f"exceeds the context limits ({limit} tokens)")
        n = p["n"]
        decoders = [_Decoder(self.engine.tokenizer, p["stop"]) for _ in range(n)]
        eos = self.engine.cfg.eos_token_id
        eos = set(eos) if isinstance(eos, list) else ({eos} if eos is not None else set())

        def stream(i: int, token: int) -> None:
            if cancelled():
                raise _Stop
            d = decoders[i]
            if d.stopped:
                return
            if token in eos:
                d.stopped = True
                delta = d.flush()
            else:
                delta = d.push(token)
            if delta and on_text is not None:
                on_text(i, delta)
            if all(x.stopped for x in decoders):
                raise _Stop

        with self.lock:
            self.requests += 1
            try:
                if p["max_tokens"]:
                    self.engine.generate([prompt_ids] * n, max_new_tokens=p["max_tokens"],
                                         temperature=p["temperature"], seed=p["seed"], stream=stream)
            except _Stop:
                pass
            except ValueError as e:  # cache limits, context length
                raise RequestError(str(e)) from e
            except RuntimeError as e:  # paged KV cache full: a request too large for this server
                if str(e).startswith("KV cache"):
                    raise RequestError(f"request exceeds the server's KV cache limits: {e}") from e
                raise
        choices = []
        for i, d in enumerate(decoders):
            tail = "" if d.stopped else d.flush()
            if tail and on_text is not None:
                on_text(i, tail)
            self.generated_tokens += len(d.ids)
            choices.append(Choice(d.text, "stop" if d.stopped else "length", len(d.ids)))
        return choices

    @staticmethod
    def _usage(prompt_ids, completion_tokens):
        return {"prompt_tokens": len(prompt_ids), "completion_tokens": completion_tokens,
                "total_tokens": len(prompt_ids) + completion_tokens}

    def _prepare(self, body: dict, chat: bool) -> tuple[list[int], dict, str]:
        if not isinstance(body, dict):
            raise RequestError("the request body must be a JSON object")
        p = self._params(body)
        if chat:
            text = self.chat_prompt(body.get("messages"))
        else:
            text = body.get("prompt")
            if isinstance(text, list) and len(text) == 1:
                text = text[0]
            if not isinstance(text, str):
                raise RequestError("`prompt` must be one string (batched prompts are not supported)")
            self._check_text(text)
        return self.encode(text), p, text

    def complete(self, body: dict, chat: bool = False) -> dict:
        ids, p, text = self._prepare(body, chat)
        choices = self.run(ids, p)
        tokens = sum(c.tokens for c in choices)
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
        head = {"id": rid, "object": "chat.completion" if chat else "text_completion",
                "created": int(time.time()), "model": self.model_name,
                "usage": self._usage(ids, tokens)}
        if chat:
            head["choices"] = [{"index": i, "message": {"role": "assistant", "content": c.text},
                                "finish_reason": c.finish_reason} for i, c in enumerate(choices)]
        else:
            echo = text if body.get("echo") else ""
            head["choices"] = [{"index": i, "text": echo + c.text, "logprobs": None,
                                "finish_reason": c.finish_reason} for i, c in enumerate(choices)]
        return head

    def stream(self, body: dict, chat: bool = False, cancelled: Callable[[], bool] = lambda: False) -> Iterator[dict]:
        """Synchronous chunk iterator (runs generation on a helper thread)."""
        import queue

        ids, p, text = self._prepare(body, chat)
        rid = f"{'chatcmpl' if chat else 'cmpl'}-{uuid.uuid4().hex[:24]}"
        created, obj = int(time.time()), "chat.completion.chunk" if chat else "text_completion"
        q: queue.Queue = queue.Queue()

        def chunk(i: int, delta: str | None, finish: str | None = None, role: bool = False) -> dict:
            if chat:
                d: dict[str, Any] = {"role": "assistant"} if role else {}
                if delta:
                    d["content"] = delta
                c = {"index": i, "delta": d, "finish_reason": finish}
            else:
                c = {"index": i, "text": delta or "", "logprobs": None, "finish_reason": finish}
            return {"id": rid, "object": obj, "created": created, "model": self.model_name, "choices": [c]}

        def work():
            try:
                choices = self.run(ids, p, on_text=lambda i, d: q.put(chunk(i, d)), cancelled=cancelled)
                for i, c in enumerate(choices):
                    q.put(chunk(i, None, c.finish_reason))
            except BaseException as e:  # noqa: BLE001  (re-raised in the consumer)
                q.put(e)
            q.put(None)

        if chat:
            for i in range(p["n"]):
                yield chunk(i, None, role=True)
        if body.get("echo") and not chat:
            yield chunk(0, text)
        threading.Thread(target=work, daemon=True).start()
        while (item := q.get()) is not None:
            if isinstance(item, BaseException):
                raise item
            yield item

    def models(self) -> dict:
        return {"object": "list", "data": [{"id": self.model_name, "object": "model", "created": self.started,
                                            "owned_by": "boilerplate"}]}

    def stats(self) -> dict:
        model = self.engine.model
        out = {"requests": self.requests, "generated_tokens": self.generated_tokens}
        if getattr(model, "expert_cache", None) is not None:
            out["expert_cache"] = model.expert_cache.stats()
        store = getattr(model, "w", None) or getattr(model, "store", None)
        if store is not None and hasattr(store, "mode_b_cache_used"):
            out["mode_b_cache_used_bytes"] = store.mode_b_cache_used
        return out


def _is_int(v) -> bool:
    return isinstance(v, int) and not isinstance(v, bool)


def _max_seq_len(engine) -> int | None:
    for obj in (engine, getattr(engine, "model", None)):
        v = getattr(obj, "max_seq_len", None)
        if _is_int(v):
            return v
    return None


def _error(message: str, status: int = 400, kind: str = "invalid_request_error"):
    from starlette.responses import JSONResponse

    headers = {"WWW-Authenticate": "Bearer"} if status == 401 else None
    return JSONResponse({"error": {"message": message, "type": kind}}, status_code=status, headers=headers)


def is_loopback(host: str) -> bool:
    if host in ("localhost", "ip6-localhost"):
        return True
    try:
        return ipaddress.ip_address(host.strip("[]")).is_loopback
    except ValueError:
        return False


def app(server: Server, api_key: str | None = None, max_body_bytes: int = MAX_BODY_BYTES,
        max_pending: int = MAX_PENDING):
    """Starlette application around ``server``.

    ``api_key``: when set, every route but ``/health`` needs ``Authorization: Bearer <api_key>``.
    ``max_body_bytes``: larger request bodies get HTTP 413 without being read in full.
    ``max_pending``: requests queued for or holding the engine; further ones get HTTP 503.
    """
    from starlette.applications import Starlette
    from starlette.requests import Request
    from starlette.responses import JSONResponse, StreamingResponse
    from starlette.routing import Route

    expected = f"Bearer {api_key}".encode() if api_key else None
    pending = [0]  # touched only from the event loop thread

    def authorized(request: Request) -> bool:
        if expected is None:
            return True
        return hmac.compare_digest(request.headers.get("authorization", "").encode(), expected)

    def guarded(fn):
        async def handle(request: Request):
            if not authorized(request):
                return _error("missing or invalid API key", 401, "authentication_error")
            return await fn(request)
        return handle

    async def read_body(request: Request) -> bytes | None:
        declared = request.headers.get("content-length")
        if declared is not None and (not declared.isdigit() or int(declared) > max_body_bytes):
            return None
        data = bytearray()
        async for part in request.stream():
            data += part
            if len(data) > max_body_bytes:
                return None
        return bytes(data)

    def endpoint(chat: bool):
        async def handle(request: Request):
            raw = await read_body(request)
            if raw is None:
                return _error(f"request body larger than {max_body_bytes} bytes", 413)
            try:
                body = json.loads(raw)
            except (json.JSONDecodeError, UnicodeDecodeError, RecursionError):
                return _error("the request body is not valid JSON")
            if pending[0] >= max_pending:
                return _error("server busy: too many queued requests, retry later", 503, "server_error")
            streaming = isinstance(body, dict) and bool(body.get("stream"))
            pending[0] += 1
            stream_owns_slot = False
            try:
                if not streaming:
                    return JSONResponse(await asyncio.to_thread(server.complete, body, chat))
                gone = threading.Event()
                chunks = server.stream(body, chat, cancelled=gone.is_set)
                first = await asyncio.to_thread(next, chunks, None)  # surfaces request errors as 400

                async def events():
                    try:
                        item = first
                        while item is not None:
                            yield f"data: {json.dumps(item)}\n\n"
                            if await request.is_disconnected():
                                gone.set()
                                return
                            item = await asyncio.to_thread(next, chunks, None)
                        yield "data: [DONE]\n\n"
                    finally:
                        gone.set()
                        pending[0] -= 1

                response = StreamingResponse(events(), media_type="text/event-stream",
                                             headers={"Cache-Control": "no-cache"})
                stream_owns_slot = True  # released when the event stream ends
                return response
            except RequestError as e:
                return _error(str(e))
            finally:
                if not stream_owns_slot:
                    pending[0] -= 1
        return guarded(handle)

    async def models(_):
        return JSONResponse(server.models())

    async def health(_):
        return JSONResponse({"status": "ok"})

    async def stats(_):
        return JSONResponse(server.stats())

    models, stats = guarded(models), guarded(stats)
    return Starlette(routes=[
        Route("/v1/completions", endpoint(False), methods=["POST"]),
        Route("/v1/chat/completions", endpoint(True), methods=["POST"]),
        Route("/v1/models", models), Route("/health", health), Route("/stats", stats),
    ])


def main(a) -> int:
    import torch
    import uvicorn

    api_key = getattr(a, "api_key", None) or os.environ.get("BOILERPLATE_API_KEY") or None
    if not api_key and not is_loopback(a.host) and not getattr(a, "insecure_no_auth", False):
        raise SystemExit(f"refusing to serve on {a.host} without an API key: pass --api-key or set "
                         "BOILERPLATE_API_KEY (or --insecure-no-auth on a trusted, isolated network)")
    torch.set_num_threads(a.threads)
    extra = {"device": a.device} if getattr(a, "device", None) else {}
    server = Server.load(a.checkpoint, name=a.name, max_seq_len=a.max_seq_len, **extra,
                         mode_b_cache_bytes=a.mode_b_cache_mib << 20,
                         expert_cache_bytes=None if a.expert_cache_mib is None else a.expert_cache_mib << 20)
    server.default_max_tokens = a.default_max_tokens
    uvicorn.run(app(server, api_key=api_key, max_body_bytes=getattr(a, "max_body_bytes", MAX_BODY_BYTES),
                    max_pending=getattr(a, "max_pending", MAX_PENDING)),
                host=a.host, port=a.port, log_level="info", server_header=False,
                limit_concurrency=getattr(a, "max_pending", MAX_PENDING) * 4, timeout_keep_alive=5)
    return 0
