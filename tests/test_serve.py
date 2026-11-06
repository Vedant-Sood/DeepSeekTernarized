"""OpenAI-compatible server on a tiny random Llama (request logic directly, plus one real HTTP round trip)."""

import json
import socket
import threading
import time

import pytest

transformers = pytest.importorskip("transformers")

from boilerplate.serve import RequestError, Server, _Decoder  # noqa: E402
from boilerplate.testing.tiny import tiny_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from boilerplate.engine.standard import StandardEngine
    from boilerplate.qat.smoke import tiny_model

    path = tmp_path_factory.mktemp("serve_llama")
    model = tiny_model("llama")
    model.config.eos_token_id = 1
    model.save_pretrained(path, safe_serialization=True)
    tiny_tokenizer(64).save(str(path / "tokenizer.json"))
    return Server(StandardEngine.load(path, max_seq_len=64), "tiny")


def _prompt():
    return "w1 w2 W3 w4"


def test_completion_matches_engine_greedy(server):
    out = server.complete({"prompt": _prompt(), "max_tokens": 8, "temperature": 0})
    ids = server.encode(_prompt())
    ref = server.engine.generate([ids], max_new_tokens=8, temperature=0.0).tokens[0]
    eos = server.engine.cfg.eos_token_id
    ref = ref[:ref.index(eos)] if eos in ref else ref
    assert out["choices"][0]["text"] == server.engine.tokenizer.decode(ref)
    assert out["usage"]["prompt_tokens"] == len(ids) and out["usage"]["completion_tokens"] == len(ref)
    assert out["object"] == "text_completion" and out["model"] == "tiny"
    echo = server.complete({"prompt": _prompt(), "max_tokens": 8, "temperature": 0, "echo": True})
    assert echo["choices"][0]["text"] == _prompt() + out["choices"][0]["text"]


def test_stream_equals_blocking_and_ends_with_finish_reason(server):
    body = {"prompt": _prompt(), "max_tokens": 10, "temperature": 0}
    full = server.complete(body)["choices"][0]
    chunks = list(server.stream(body))
    assert "".join(c["choices"][0]["text"] for c in chunks) == full["text"]
    assert chunks[-1]["choices"][0]["finish_reason"] == full["finish_reason"]
    assert all(c["choices"][0]["finish_reason"] is None for c in chunks[:-1])


def test_stop_string_truncates_and_finishes(server):
    full = server.complete({"prompt": _prompt(), "max_tokens": 12, "temperature": 0})["choices"][0]["text"]
    words = full.split()
    if len(words) < 3:
        pytest.skip("greedy output too short to pick a stop word")
    stop = words[2]
    expected = full[:full.find(stop)]
    out = server.complete({"prompt": _prompt(), "max_tokens": 12, "temperature": 0, "stop": stop})["choices"][0]
    assert out["text"] == expected and out["finish_reason"] == "stop"
    streamed = "".join(c["choices"][0]["text"] for c in server.stream(
        {"prompt": _prompt(), "max_tokens": 12, "temperature": 0, "stop": [stop]}))
    assert streamed == expected


def test_sampled_choices_are_seeded(server):
    body = {"prompt": _prompt(), "max_tokens": 6, "temperature": 1.0, "seed": 7, "n": 3}
    a, b = server.complete(body), server.complete(body)
    assert [c["text"] for c in a["choices"]] == [c["text"] for c in b["choices"]]
    assert len(a["choices"]) == 3


@pytest.mark.parametrize("body,match", [
    ({"prompt": "w1", "top_p": 0.5}, "top_p"),
    ({"prompt": "w1", "n": 2, "temperature": 0}, "temperature"),
    ({"prompt": "w1", "max_tokens": -1}, "max_tokens"),
    ({"prompt": ["w1", "w2"]}, "one string"),
    ({"prompt": "w1", "stop": [""]}, "stop"),
    ({"prompt": "w1", "max_tokens": 1000}, "limits"),
])
def test_invalid_requests(server, body, match):
    with pytest.raises(RequestError, match=match):
        server.complete(body)


def test_chat_template_and_fallback(server):
    msgs = [{"role": "system", "content": "w1"}, {"role": "user", "content": "w2 w3"}]
    assert server.chat_prompt(msgs) == "system: w1\nuser: w2 w3\nassistant:"
    templated = Server(server.engine, "tiny", chat_template=(
        "{% for m in messages %}<{{ m.role }}>{{ m.content }}{% endfor %}{% if add_generation_prompt %}<assistant>{% endif %}"))
    assert templated.chat_prompt(msgs) == "<system>w1<user>w2 w3<assistant>"
    out = server.complete({"messages": msgs, "max_tokens": 4, "temperature": 0}, chat=True)
    assert out["object"] == "chat.completion" and out["choices"][0]["message"]["role"] == "assistant"
    chunks = list(server.stream({"messages": msgs, "max_tokens": 4, "temperature": 0}, chat=True))
    assert chunks[0]["choices"][0]["delta"] == {"role": "assistant"}
    assert "".join(c["choices"][0]["delta"].get("content", "") for c in chunks) == out["choices"][0]["message"]["content"]
    with pytest.raises(RequestError):
        server.chat_prompt([{"role": "wizard", "content": "x"}])


class _CharTok:
    def __init__(self, pieces):
        self.pieces = pieces

    def decode(self, ids):
        return "".join(self.pieces[i] for i in ids)


def test_decoder_holds_back_partial_stop_and_utf8():
    d = _Decoder(_CharTok(["ab", "c", "X", "Y", "�"]), stop=["XY"])
    assert d.push(0) == "ab" and d.push(1) == "c"
    assert d.push(2) == ""  # could be the start of "XY"
    assert d.push(3) == "" and d.stopped and d.text == "abc"
    d = _Decoder(_CharTok(["ab", "�"]), stop=[])
    assert d.push(0) == "ab" and d.push(1) == ""  # incomplete character held back
    assert d.flush() == "�"


def test_http_round_trip(server):
    uvicorn = pytest.importorskip("uvicorn")
    requests = pytest.importorskip("requests")
    from boilerplate.serve import app

    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    srv = uvicorn.Server(uvicorn.Config(app(server), host="127.0.0.1", port=port, log_level="warning"))
    thread = threading.Thread(target=srv.run, daemon=True)
    thread.start()
    base = f"http://127.0.0.1:{port}"
    try:
        for _ in range(100):
            if srv.started:
                break
            time.sleep(0.05)
        assert requests.get(f"{base}/health").json() == {"status": "ok"}
        assert requests.get(f"{base}/v1/models").json()["data"][0]["id"] == "tiny"
        body = {"prompt": _prompt(), "max_tokens": 6, "temperature": 0}
        full = requests.post(f"{base}/v1/completions", json=body).json()["choices"][0]["text"]
        r = requests.post(f"{base}/v1/completions", json={**body, "stream": True}, stream=True)
        assert r.headers["content-type"].startswith("text/event-stream")
        events = [line[6:] for line in r.iter_lines(decode_unicode=True) if line.startswith("data: ")]
        assert events[-1] == "[DONE]"
        assert "".join(json.loads(e)["choices"][0]["text"] for e in events[:-1]) == full
        bad = requests.post(f"{base}/v1/completions", json={"prompt": "w1", "top_p": 0.1})
        assert bad.status_code == 400 and "top_p" in bad.json()["error"]["message"]
        bad = requests.post(f"{base}/v1/chat/completions", json={"messages": [], "stream": True})
        assert bad.status_code == 400
        assert requests.get(f"{base}/stats").json()["requests"] >= 1
    finally:
        srv.should_exit = True
        thread.join(timeout=10)


def test_chat_template_is_sandboxed(server):
    # A malicious tokenizer_config.json chat template must not reach Python internals.
    evil = Server(server.engine, "tiny", chat_template=(
        "{{ messages.__class__.__mro__[1].__subclasses__() }}"))
    with pytest.raises(RequestError, match="unsafe|template"):
        evil.chat_prompt([{"role": "user", "content": "w1"}])
    mutating = Server(server.engine, "tiny", chat_template="{{ messages.append(1) }}")
    with pytest.raises(RequestError):
        mutating.chat_prompt([{"role": "user", "content": "w1"}])
    raising = Server(server.engine, "tiny", chat_template="{{ raise_exception('no system role') }}")
    with pytest.raises(RequestError, match="no system role"):
        raising.chat_prompt([{"role": "user", "content": "w1"}])


@pytest.mark.parametrize("body,match", [
    ({"prompt": "w1", "max_tokens": True}, "max_tokens"),
    ({"prompt": "w1", "n": True}, "`n`"),
    ({"prompt": "w1", "seed": "7"}, "seed"),
    ({"prompt": "w1", "seed": -1}, "seed"),
    ({"prompt": "w1", "stop": 5}, "stop"),
    ({"prompt": "w1", "stop": "x" * 1000}, "stop"),
    ({"prompt": "w1", "temperature": False}, "temperature"),
])
def test_rejects_malformed_parameters(server, body, match):
    with pytest.raises(RequestError, match=match):
        server.complete(body)


def test_prompt_length_is_bounded_before_tokenizing(server):
    small = Server(server.engine, "tiny", max_prompt_chars=16)
    with pytest.raises(RequestError, match="longer than"):
        small.complete({"prompt": "w1 " * 100, "max_tokens": 1})
    with pytest.raises(RequestError, match="longer than"):
        small.complete({"messages": [{"role": "user", "content": "w1 " * 100}], "max_tokens": 1}, chat=True)


def test_http_auth_body_limit_and_busy(server):
    pytest.importorskip("starlette")
    from starlette.testclient import TestClient

    from boilerplate.serve import app, is_loopback

    client = TestClient(app(server, api_key="s3cret", max_body_bytes=256))
    assert client.get("/health").status_code == 200  # liveness stays open
    assert client.get("/v1/models").status_code == 401
    assert client.get("/stats", headers={"Authorization": "Bearer wrong"}).status_code == 401
    r = client.post("/v1/completions", json={"prompt": "w1", "max_tokens": 1})
    assert r.status_code == 401 and r.headers["www-authenticate"] == "Bearer"
    auth = {"Authorization": "Bearer s3cret"}
    assert client.get("/v1/models", headers=auth).status_code == 200
    ok = client.post("/v1/completions", json={"prompt": "w1", "max_tokens": 2, "temperature": 0}, headers=auth)
    assert ok.status_code == 200
    big = client.post("/v1/completions", json={"prompt": "w1 " * 200, "max_tokens": 1}, headers=auth)
    assert big.status_code == 413
    busy = TestClient(app(server, max_pending=0))
    assert busy.post("/v1/completions", json={"prompt": "w1", "max_tokens": 1}).status_code == 503
    assert is_loopback("127.0.0.1") and is_loopback("::1") and is_loopback("localhost")
    assert not is_loopback("0.0.0.0") and not is_loopback("192.168.1.5")


def test_main_refuses_public_bind_without_key(monkeypatch):
    import argparse

    from boilerplate import serve

    monkeypatch.delenv("BOILERPLATE_API_KEY", raising=False)
    a = argparse.Namespace(host="0.0.0.0", api_key=None, insecure_no_auth=False)
    with pytest.raises(SystemExit, match="without an API key"):
        serve.main(a)
