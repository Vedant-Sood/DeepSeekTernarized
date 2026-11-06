"""boilerplate serve and the in-process backend, with a stub engine (no weights needed)."""

from __future__ import annotations

import json
import threading
import urllib.error
import urllib.request
from http.server import ThreadingHTTPServer
from types import SimpleNamespace

import pytest

from boilerplate.agent.backends import BackendError, LocalEngineBackend, OpenAIBackend
from boilerplate.agent.chat_format import ChatTemplate
from boilerplate.agent.config import BackendConfig
from boilerplate.engine.engine import GenerationResult
from boilerplate.serve import ServeState, make_handler


class CharTokenizer:
    def encode(self, text, add_special_tokens=True):
        return SimpleNamespace(ids=[ord(c) for c in text])

    def decode(self, ids):
        return "".join(chr(i) for i in ids)


class StubEngine:
    """Replies with the next scripted string, token by token, honouring the stream callback."""

    def __init__(self, replies):
        self.replies = list(replies)
        self.tokenizer = CharTokenizer()
        self.prompts = []

    def generate(self, prompts, max_new_tokens=32, temperature=0.0, seed=None, eos_token_ids=None, stream=None):
        self.prompts.append(self.tokenizer.decode(prompts[0]))
        reply = self.replies.pop(0)
        out = []
        for t in [ord(c) for c in reply][:max_new_tokens]:
            out.append(t)
            if stream:
                stream(0, t)
        return GenerationResult([out], [len(out) < max_new_tokens])


CALL = '<tool_call>{"name": "read_file", "arguments": {"path": "a.py"}}</tool_call> trailing junk'


def test_local_backend_text_protocol():
    eng = StubEngine(["Let me read it.\n" + CALL])
    b = LocalEngineBackend(BackendConfig(kind="local", max_tokens=500, max_seq_len=100_000), engine=eng,
                           template=ChatTemplate())
    seen = []
    turn = b.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}],
                  [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {}}}],
                  on_text=seen.append)
    assert turn.text == "Let me read it." and turn.tool_calls[0].arguments == {"path": "a.py"}
    assert "trailing junk" not in "".join(seen)  # generation stopped at </tool_call>
    assert "## read_file" in eng.prompts[0] and eng.prompts[0].endswith("<|assistant|>\n")
    tiny = LocalEngineBackend(BackendConfig(kind="local", max_seq_len=10), engine=StubEngine(["x"]),
                              template=ChatTemplate())
    with pytest.raises(BackendError, match="context"):
        tiny.chat([{"role": "user", "content": "a long prompt"}], [])


@pytest.fixture
def server():
    eng = StubEngine([])
    state = ServeState(eng, ChatTemplate(), "stub", max_seq_len=100_000, max_tokens_cap=1000, api_key="s3cret-key")
    httpd = ThreadingHTTPServer(("127.0.0.1", 0), make_handler(state))
    threading.Thread(target=httpd.serve_forever, daemon=True).start()
    yield eng, f"http://127.0.0.1:{httpd.server_address[1]}"
    httpd.shutdown()


def _post(url, body, key="s3cret-key"):
    req = urllib.request.Request(url, data=json.dumps(body).encode(), method="POST",
                                 headers={"Content-Type": "application/json", "Authorization": f"Bearer {key}"})
    return urllib.request.urlopen(req, timeout=10)


def test_serve_auth_and_validation(server):
    eng, base = server
    with pytest.raises(urllib.error.HTTPError) as e:
        _post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "x"}]}, key="wrong")
    assert e.value.code == 401
    for bad in ({}, {"messages": []}, {"messages": [{"role": "evil", "content": "x"}]},
                {"messages": [{"role": "user", "content": "x"}], "max_tokens": -1},
                {"messages": [{"role": "user", "content": "x"}], "temperature": 9}):
        with pytest.raises(urllib.error.HTTPError) as e:
            _post(base + "/v1/chat/completions", bad)
        assert e.value.code == 400
    req = urllib.request.Request(base + "/v1/models", headers={"Authorization": "Bearer s3cret-key"})
    assert json.load(urllib.request.urlopen(req))["data"][0]["id"] == "stub"


def test_serve_non_streaming(server):
    eng, base = server
    eng.replies.append("plain answer")
    data = json.load(_post(base + "/v1/chat/completions", {"messages": [{"role": "user", "content": "hi"}]}))
    assert data["choices"][0]["message"]["content"] == "plain answer"
    assert data["choices"][0]["finish_reason"] == "stop"


def test_agent_backend_round_trip_through_serve(server, monkeypatch):
    """OpenAI client (native tool calls, streaming) -> serve -> text protocol on the engine."""
    eng, base = server
    eng.replies.append("Checking.\n" + CALL)
    monkeypatch.setenv("SERVE_KEY", "s3cret-key")
    b = OpenAIBackend(BackendConfig(endpoint=base + "/v1", model="stub", api_key_env="SERVE_KEY"))
    seen = []
    tools = [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {}}}]
    turn = b.chat([{"role": "user", "content": "go"}], tools, on_text=seen.append)
    assert "".join(seen).strip() == "Checking." and turn.text.strip() == "Checking."
    assert turn.tool_calls[0].name == "read_file" and turn.tool_calls[0].arguments == {"path": "a.py"}
    assert turn.finish_reason == "tool_calls"
    # the follow-up request carries the tool result back; the engine sees it in the prompt
    eng.replies.append("Done.")
    msgs = [{"role": "user", "content": "go"}, turn.to_message(),
            {"role": "tool", "tool_call_id": turn.tool_calls[0].id, "name": "read_file", "content": "FILE BODY"}]
    turn2 = b.chat(msgs, tools)
    assert turn2.text == "Done." and "FILE BODY" in eng.prompts[-1] and '"name": "read_file"' in eng.prompts[-1]


def test_serve_refuses_public_bind_without_key(monkeypatch):
    from boilerplate.serve import serve

    with pytest.raises(SystemExit, match="non-loopback"):
        serve("unused", host="0.0.0.0", engine=StubEngine([]), template=ChatTemplate())
