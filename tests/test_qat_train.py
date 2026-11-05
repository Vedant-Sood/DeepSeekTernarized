"""QAT trainer end to end on tiny random models (a few CPU steps; no pretrained model is trained)."""

import json

import numpy as np
import pytest
import torch

transformers = pytest.importorskip("transformers")

from boilerplate.qat.data import Sampler, TokenWindows, write_tokens  # noqa: E402
from boilerplate.qat.train import RunConfig, train  # noqa: E402


def _source(path, kind):
    from boilerplate.qat.smoke import tiny_model

    model = tiny_model(kind)
    model.save_pretrained(path, safe_serialization=True)
    return model


def _tokens(path, n=4000, vocab=64):
    ids = np.random.default_rng(0).integers(0, vocab, n)
    return write_tokens(ids, path, vocab_size=vocab)


def _cfg(tmp_path, kind="llama", **kw):
    src = tmp_path / f"src_{kind}"
    _source(src, kind)
    toks = _tokens(tmp_path / "train.bin")
    base = dict(source=str(src), train_tokens=str(toks), out_dir=str(tmp_path / f"run_{kind}"),
                token_budget=16 * 2 * 6, seq_len=16, micro_batch=2, grad_accum=1, lr_warmup_steps=2,
                quant_warmup_steps=3, group=16, master_dtype="fp32", log_every=1, checkpoint_every=2,
                routing_every=2, probe_windows=2, device="cpu")
    base.update(kw)
    return RunConfig(**base)


def test_sampler_resume_and_rank_split():
    a = Sampler(10, seed=3)
    first = a.next(7)
    state = a.state()
    rest = a.next(8)
    b = Sampler(10, seed=3)
    b.load_state(state)
    assert b.next(8) == rest
    r0, r1 = Sampler(10, seed=3, rank=0, world=2), Sampler(10, seed=3, rank=1, world=2)
    assert not set(r0.next(5)) & set(r1.next(5))
    assert sorted(first[:7]) == sorted(set(first[:7]))


def test_token_windows(tmp_path):
    path = write_tokens([[1, 2, 3], [4, 5, 6, 7]], tmp_path / "t.bin", vocab_size=10)
    w = TokenWindows(path, 3)
    assert len(w) == 2 and w.window(1).tolist() == [4, 5, 6, 7]
    assert json.loads(path.with_suffix(".json").read_text())["dtype"] == "uint16"


def test_approval_gate(tmp_path):
    cfg = _cfg(tmp_path, token_budget=16 * 2 * 100)
    with pytest.raises(SystemExit, match="approval"):
        train(cfg, log=lambda *_: None)


def test_train_export_parity_and_exact_resume(tmp_path):
    from boilerplate.engine.standard import StandardEngine
    from boilerplate.qat.distributed import set_quantization_level
    from boilerplate.qat.model import load_for_qat

    cfg = _cfg(tmp_path)
    result = train(cfg, log=lambda *_: None)
    assert result["steps"] == 6 and "mode_d" in result
    log = [json.loads(line) for line in open(f"{cfg.out_dir}/log.rank0.jsonl")]
    assert [r["step"] for r in log] == list(range(1, 7)) and log[-1]["lambda"] == 1.0
    manifest = json.loads(open(f"{result['mode_d']}/manifest.json").read())
    assert {e["mode"] for e in manifest["tensors"].values()} >= {"A", "D"}

    # export parity: the training graph at lambda = 1 vs the engine on the exported Mode D weights
    model, _ = load_for_qat(cfg.source, dtype=torch.float32, group=16)
    from boilerplate.qat.checkpoint import load_checkpoint
    from boilerplate.qat.train import _optimizer

    load_checkpoint(f"{cfg.out_dir}/checkpoint.pt", model, _optimizer([p for p in model.parameters() if p.requires_grad], cfg))
    set_quantization_level(model, 1.0)
    ids = torch.randint(0, 64, (1, 12), generator=torch.Generator().manual_seed(1))
    with torch.no_grad():
        expected = model.eval()(ids).logits
    engine = StandardEngine.load(result["mode_d"], ternary_backend="torch")
    actual = engine.trace(ids).logits
    assert torch.allclose(actual, expected, atol=2e-3, rtol=2e-3), (actual - expected).abs().max()

    # resume: 3 steps, stop, continue to 6 == uninterrupted 6 (same final checkpoint weights)
    cfg2 = _cfg(tmp_path, out_dir=str(tmp_path / "run_resume"), max_steps=3)
    train(cfg2, log=lambda *_: None)
    cfg2.max_steps = None
    train(cfg2, log=lambda *_: None)
    a = torch.load(f"{cfg.out_dir}/checkpoint.pt", weights_only=False)["model"]
    b = torch.load(f"{cfg2.out_dir}/checkpoint.pt", weights_only=False)["model"]
    assert all(torch.equal(a[k], b[k]) for k in a if isinstance(a[k], torch.Tensor))


def test_olmoe_with_offline_teacher(tmp_path):
    from boilerplate.qat.teacher import TeacherStore, precompute

    cfg = _cfg(tmp_path, "olmoe", token_budget=16 * 2 * 4)
    teacher_model = transformers.AutoModelForCausalLM.from_pretrained(cfg.source)
    windows = TokenWindows(cfg.train_tokens, cfg.seq_len)
    tdir = precompute(teacher_model, windows, tmp_path / "teacher", list(range(len(windows))), k=8,
                      per_shard=64, log=lambda *_: None)
    store = TeacherStore(tdir)
    t = store.get([3, 70])
    assert t["indices"].shape == (2, 16, 8) and t["log_probs"].dtype == torch.float32
    cfg.teacher_dir = str(tdir)
    result = train(cfg, log=lambda *_: None)
    log = [json.loads(line) for line in open(f"{cfg.out_dir}/log.rank0.jsonl")]
    assert "routing" in log[-1] and len(log[-1]["routing"]) == 2
    manifest = json.loads(open(f"{result['mode_d']}/manifest.json").read())
    assert manifest["tensors"]["model.layers.0.mlp.experts.0.gate_proj"]["mode"] == "D"
    assert manifest["tensors"]["model.layers.0.mlp.gate"]["mode"] == "A"


def test_cast_checkpoint_f32_to_bf16(tmp_path):
    from safetensors.torch import load_file

    from boilerplate.checkpoint.cast import cast_checkpoint

    src = tmp_path / "f32"
    _source(src, "olmoe")
    out = cast_checkpoint(src, tmp_path / "bf16", "bf16", log=lambda *_: None)
    a, b = load_file(src / "model.safetensors"), load_file(out / "model.safetensors")
    assert a.keys() == b.keys()
    for k in a:
        assert b[k].dtype == torch.bfloat16 and torch.equal(b[k], a[k].to(torch.bfloat16))
    rec = json.loads((out / "cast.json").read_text())
    assert rec["shards"]["model.safetensors"]["tensors"] == len(a)
    assert json.loads((out / "config.json").read_text())["torch_dtype"] == "bfloat16"
