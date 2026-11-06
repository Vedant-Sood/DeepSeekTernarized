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

    cfg = _cfg(tmp_path, export_at_tokens=[16 * 2 * 2])
    result = train(cfg, log=lambda *_: None)
    assert result["steps"] == 6 and "mode_d" in result
    assert (tmp_path / "run_llama" / "mode_d_tq1_64tok" / "manifest.json").exists()
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


def test_qat_estimate(tmp_path):
    from boilerplate.qat.estimate import estimate, param_counts

    cfg = _cfg(tmp_path, "olmoe")
    p = param_counts(cfg.source)
    assert p["experts"] == 4 and p["top_k"] == 2 and 0 < p["expert"] < p["total"]
    assert p["active"] == p["total"] - p["expert"] // 2
    e = estimate(cfg, world=2, gpu="h100", mfu=0.5)
    assert e["train_flops"] == 6 * p["active"] * e["tokens"] and e["fits"]
    measured = estimate(cfg, world=2, tok_s=100.0, price_per_gpu_hour=2.0)
    assert measured["wall_clock_hours"] == pytest.approx(e["tokens"] / 100 / 3600)
    assert measured["cost"] == pytest.approx(measured["gpu_hours"] * 2.0)


def test_mode_d_export_keeps_source_dtype_for_mode_a_tensors(tmp_path):
    from safetensors import safe_open

    from boilerplate.checkpoint.cast import cast_checkpoint
    from boilerplate.qat.export import export_mode_d
    from boilerplate.qat.model import load_for_qat

    src = tmp_path / "f32"
    _source(src, "llama")
    bf16 = cast_checkpoint(src, tmp_path / "bf16", "bf16", log=lambda *_: None)
    model, _ = load_for_qat(str(bf16), dtype=torch.float32, group=16)  # FP32 master copy
    out = export_mode_d(model, bf16, tmp_path / "mode_d", log=lambda *_: None)
    manifest = json.loads((out / "manifest.json").read_text())
    shard = next(out.glob("*.safetensors"))
    with safe_open(str(shard), "pt") as h:
        dtypes = {k: h.get_slice(k).get_dtype() for k in h.keys()}
    embed = [k for k in dtypes if "embed_tokens" in k]
    assert embed and all(dtypes[k] == "BF16" for k in embed)
    assert manifest["tensors"]["model.layers.0.mlp.down_proj"]["mode"] == "D"


def test_streaming_tokenizer_txt_and_jsonl(tmp_path):
    from boilerplate.qat.data import tokenize_text_file
    from boilerplate.testing.tiny import tiny_tokenizer

    tok = tiny_tokenizer(64)
    tok.save(str(tmp_path / "tok.json"))
    lines = [" ".join(f"w{(i * 7 + j) % 15}" for j in range(i % 9 + 1)) for i in range(200)]
    (tmp_path / "c.txt").write_text("\n".join(lines) + "\n")
    out = tokenize_text_file(tmp_path / "c.txt", tmp_path / "tok.json", tmp_path / "c.bin", chunk_chars=37)
    whole = tok.encode("\n".join(lines) + "\n", add_special_tokens=False).ids
    assert TokenWindows(out, 1).tokens.tolist() == whole
    (tmp_path / "d.jsonl").write_text("".join(json.dumps({"text": t}) + "\n" for t in lines[:5]))
    out = tokenize_text_file(tmp_path / "d.jsonl", tmp_path / "tok.json", tmp_path / "d.bin")
    meta = json.loads(out.with_suffix(".json").read_text())
    assert meta["documents"] == 5 and meta["eos_id"] == 1  # "</s>"
    want = [t for line in lines[:5] for t in tok.encode(line, add_special_tokens=False).ids + [1]]
    assert TokenWindows(out, 1).tokens.tolist() == want
    assert meta["sha256"] == __import__("hashlib").sha256(np.asarray(want, dtype=np.uint16).tobytes()).hexdigest()


def test_prepare_data_local_parquet_and_jsonl_gz(tmp_path):
    import gzip

    import pyarrow as pa
    import pyarrow.parquet as pq

    from boilerplate.qat.prepare_data import prepare

    src = tmp_path / "src"
    src.mkdir()
    for i in range(3):
        pq.write_table(pa.table({"text": [f"p{i} doc {j} " * 5 for j in range(20)]}), src / f"{i:03d}.parquet")
    for i in range(2):
        with gzip.open(src / f"w{i}.json.gz", "wt") as f:
            for j in range(20):
                f.write(json.dumps({"text": f"g{i} doc {j} " * 5}) + "\n")
    out = prepare(tmp_path / "data", [f"{src}:*.parquet:3", f"{src}:w*.json.gz:1"], train_chars=2400,
                  heldout_chars=400, log=lambda *_: None)
    train = [json.loads(l)["text"] for l in open(out / "train.jsonl")]
    held = [json.loads(l)["text"] for l in open(out / "heldout.jsonl")]
    assert not set(train) & set(held)
    assert all(t.startswith(("p2", "g1")) for t in held)  # last file of each source
    assert not any(t.startswith(("p2", "g1")) for t in train)
    rec = json.loads((out / "data.json").read_text())
    p, g = rec["sources"]
    assert p["train_chars"] >= 1800 and g["train_chars"] >= 600 and g["train_chars"] < 1800
    assert rec["train_docs"] == len(train) and rec["heldout_docs"] == len(held)


def test_online_teacher_matches_offline(tmp_path):
    from boilerplate.qat.teacher import precompute

    off = _cfg(tmp_path, "llama", token_budget=16 * 2 * 3, out_dir=str(tmp_path / "off"))
    windows = TokenWindows(off.train_tokens, off.seq_len)
    teacher = transformers.AutoModelForCausalLM.from_pretrained(off.source)
    off.teacher_dir = str(precompute(teacher, windows, tmp_path / "t", list(range(len(windows))), k=8,
                                     log=lambda *_: None))
    train(off, log=lambda *_: None)
    on = _cfg(tmp_path, "llama", token_budget=16 * 2 * 3, out_dir=str(tmp_path / "on"),
              teacher_model=off.source, teacher_k=8, teacher_dtype="fp32")
    train(on, log=lambda *_: None)
    a = [json.loads(l)["loss"] for l in open(tmp_path / "off" / "log.rank0.jsonl")]
    b = [json.loads(l)["loss"] for l in open(tmp_path / "on" / "log.rank0.jsonl")]
    assert a == pytest.approx(b, rel=2e-3)  # offline stores fp16 log-probs
    bad = _cfg(tmp_path, "llama", out_dir=str(tmp_path / "bad"), teacher_dir=off.teacher_dir, seq_len=8)
    with pytest.raises(SystemExit, match="teacher"):
        train(bad, log=lambda *_: None)


def test_sorted_expert_dispatch_matches_per_expert_scan():
    from boilerplate.qat.model import Expert, Experts

    torch.manual_seed(0)
    E, H, inter, T, k = 8, 32, 16, 50, 2
    experts = Experts([Expert(H, inter) for _ in range(E)])
    h = torch.randn(T, H, requires_grad=True)
    idx = torch.stack([torch.randperm(E)[:k] for _ in range(T)])
    idx[idx == 7] = 6  # one expert unused
    w = torch.rand(T, k)
    ref = torch.zeros_like(h)
    for e, expert in enumerate(experts):
        token, slot = torch.where(idx == e)
        if token.numel():
            ref = ref.index_add(0, token, expert(h[token]) * w[token, slot, None])
    out = experts(h, idx, w)
    assert torch.allclose(out, ref, atol=1e-6)
    g1, = torch.autograd.grad(out.sum(), h)
    g2, = torch.autograd.grad(ref.sum(), h)
    assert torch.allclose(g1, g2, atol=1e-6)
