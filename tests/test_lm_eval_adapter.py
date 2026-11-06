"""lm-evaluation-harness adapter: scores checked against transformers on a tiny random Llama."""

from types import SimpleNamespace

import pytest
import torch

transformers = pytest.importorskip("transformers")

from boilerplate.evals.lm_eval_adapter import Scorer  # noqa: E402
from boilerplate.testing.tiny import tiny_tokenizer  # noqa: E402


@pytest.fixture(scope="module")
def setup(tmp_path_factory):
    from boilerplate.engine.standard import StandardEngine
    from boilerplate.qat.smoke import tiny_model

    path = tmp_path_factory.mktemp("lmeval_llama")
    model = tiny_model("llama")
    model.config.bos_token_id, model.config.eos_token_id = 0, 1
    model.save_pretrained(path, safe_serialization=True)
    tiny_tokenizer(64).save(str(path / "tokenizer.json"))
    engine = StandardEngine.load(path, max_seq_len=64)
    return model.eval(), engine, path


def _hf_logprob(model, ctx, target):
    ids = torch.tensor([ctx + target])
    with torch.no_grad():
        lp = torch.log_softmax(model(ids).logits[0, :-1].float(), -1)
    picked = lp[len(ctx) - 1:].gather(1, torch.tensor(target)[:, None])
    return float(picked.sum()), bool((lp[len(ctx) - 1:].argmax(-1) == torch.tensor(target)).all())


def test_loglikelihood_matches_transformers(setup):
    model, engine, _ = setup
    s = Scorer(engine)
    assert s.prefix == 0
    ctx, cont = s.encode("w1 w2 W3"), s.encode("w4 w5")
    got, greedy = s.loglikelihood("w1 w2 W3 ", "w4 w5")  # trailing space moves to the continuation
    want, want_greedy = _hf_logprob(model, ctx, cont)
    assert got == pytest.approx(want, abs=1e-4) and greedy == want_greedy
    empty, _ = s.loglikelihood("", "w4 w5")
    assert empty == pytest.approx(_hf_logprob(model, [0], cont)[0], abs=1e-4)
    # greedy flag: the argmax continuation is greedy by construction
    nxt = int(torch.argmax(model(torch.tensor([ctx])).logits[0, -1]))
    word = engine.tokenizer.id_to_token(nxt)
    if nxt > 3 and word.isalnum():  # a plain word the tokenizer maps back to the same id
        assert s.loglikelihood("w1 w2 W3", " " + word)[1]


def test_rolling_windows_equal_full_sum_and_truncate(setup):
    model, engine, _ = setup
    text = " ".join(f"w{i % 9}" for i in range(20))
    s = Scorer(engine)
    tokens = [0] + s.encode(text)
    full = s.loglikelihood_rolling(text)
    want = _hf_logprob(model, tokens[:1], tokens[1:])[0]
    assert full == pytest.approx(want, abs=1e-3)
    small = Scorer(engine, max_length=8)
    manual = 0.0
    for start in range(1, len(tokens), 8):
        end = min(start + 8, len(tokens))
        ctx_start = max(0, end - 1 - 8)
        manual += _hf_logprob(model, tokens[ctx_start:start], tokens[start:end])[0]
    assert small.loglikelihood_rolling(text) == pytest.approx(manual, abs=1e-3)


def test_generate_until_stops(setup):
    _, engine, _ = setup
    s = Scorer(engine)
    free = s.generate_until("w1 w2", {"max_gen_toks": 10})
    words = free.split()
    if len(words) >= 2:
        cut = s.generate_until("w1 w2", {"until": [words[1]], "max_gen_toks": 10})
        assert cut == free[:free.find(words[1])]


def test_harness_lm_class(setup):
    pytest.importorskip("lm_eval")
    from boilerplate.evals.lm_eval_adapter import BoilerplateLM

    _, engine, path = setup
    lm = BoilerplateLM(engine=engine)
    reqs = [SimpleNamespace(args=("w1 w2", " w3")), SimpleNamespace(args=("", "w3"))]
    out = lm.loglikelihood(reqs, disable_tqdm=True)
    assert len(out) == 2 and all(isinstance(x[0], float) and isinstance(x[1], bool) for x in out)
    assert lm.loglikelihood_rolling([SimpleNamespace(args=("w1 w2 w3",))], disable_tqdm=True)[0] < 0
    assert isinstance(lm.generate_until([SimpleNamespace(args=("w1", {"max_gen_toks": 3}))], disable_tqdm=True)[0], str)
    from lm_eval.api.registry import get_model

    assert get_model("boilerplate") is BoilerplateLM
    loaded = get_model("boilerplate").create_from_arg_string(f"path={path},max_seq_len=64")
    assert loaded.loglikelihood(reqs[:1], disable_tqdm=True)[0][0] == pytest.approx(out[0][0], abs=1e-5)
