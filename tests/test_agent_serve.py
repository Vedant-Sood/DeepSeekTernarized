"""The agent's in-process backend and its compatibility with ``boilerplate serve``, using a stub
engine (no weights needed)."""

from __future__ import annotations

import json
from types import SimpleNamespace

import pytest

from boilerplate.agent.backends import BackendError, LocalEngineBackend, OpenAIBackend
from boilerplate.agent.chat_format import ChatTemplate
from boilerplate.agent.config import BackendConfig
from boilerplate.engine.engine import GenerationResult
from boilerplate.serve import Server


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
        self.cfg = SimpleNamespace(eos_token_id=None)
        self.max_seq_len = 100_000
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
TOOLS = [{"type": "function", "function": {"name": "read_file", "description": "d", "parameters": {}}}]


def test_local_backend_text_protocol():
    eng = StubEngine(["Let me read it.\n" + CALL])
    b = LocalEngineBackend(BackendConfig(kind="local", max_tokens=500, max_seq_len=100_000), engine=eng,
                           template=ChatTemplate())
    seen = []
    turn = b.chat([{"role": "system", "content": "sys"}, {"role": "user", "content": "go"}], TOOLS,
                  on_text=seen.append)
    assert turn.text == "Let me read it." and turn.tool_calls[0].arguments == {"path": "a.py"}
    assert "trailing junk" not in "".join(seen)  # generation stopped at </tool_call>
    assert "## read_file" in eng.prompts[0] and eng.prompts[0].endswith("<|assistant|>\n")
    tiny = LocalEngineBackend(BackendConfig(kind="local", max_seq_len=10), engine=StubEngine(["x"]),
                              template=ChatTemplate())
    with pytest.raises(BackendError, match="context"):
        tiny.chat([{"role": "user", "content": "a long prompt"}], [])


class _Resp:
    """Feeds ``Server.stream`` output to ``OpenAIBackend._read_stream`` as SSE lines."""

    def __init__(self, chunks):
        self.lines = [f"data: {json.dumps(c)}\n".encode() for c in chunks] + [b"data: [DONE]\n"]

    def __iter__(self):
        return iter(self.lines)


def test_agent_text_protocol_through_boilerplate_serve():
    """The agent's request is accepted by the engine's server, which stops at </tool_call>, and the
    streamed reply parses back into a tool call; the tool result reaches the engine's prompt."""
    eng = StubEngine(["Checking.\n" + CALL, "Done."])
    server = Server(eng, "stub")
    b = OpenAIBackend(BackendConfig(endpoint="http://127.0.0.1:1/v1", model="stub", tool_protocol="text"))
    payload = b._payload([{"role": "user", "content": "go"}], TOOLS)
    turn = b._read_stream(_Resp(list(server.stream(payload, chat=True))), None)
    # chat() completes an unterminated call cut by the stop string; do the same here
    from boilerplate.agent.chat_format import parse_text_tool_calls

    prose, calls = parse_text_tool_calls(turn.text)
    assert prose == "Checking." and calls[0].name == "read_file" and calls[0].arguments == {"path": "a.py"}
    assert "trailing junk" not in turn.text
    msgs = [{"role": "user", "content": "go"},
            {"role": "assistant", "content": prose, "tool_calls": [calls[0].to_message()]},
            {"role": "tool", "tool_call_id": calls[0].id, "name": "read_file", "content": "FILE BODY"}]
    out = server.complete(b._payload(msgs, TOOLS) | {"stream": False}, chat=True)
    assert out["choices"][0]["message"]["content"] == "Done."
    assert "FILE BODY" in eng.prompts[-1] and '"name": "read_file"' in eng.prompts[-1]
