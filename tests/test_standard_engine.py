"""Standard decoder correctness against tiny random Hugging Face models."""

import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from boilerplate.engine import Engine
from boilerplate.engine.standard import StandardEngine
from boilerplate.engine.weights import WeightStore
from boilerplate.format.export import export
from boilerplate.format.policy import Policy, Rule


def _make_model(path, kind):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(71)
    if kind == "llama":
        config = transformers.LlamaConfig(vocab_size=97, hidden_size=24, intermediate_size=48,
                                          num_hidden_layers=2, num_attention_heads=4,
                                          num_key_value_heads=2, tie_word_embeddings=True,
                                          max_position_embeddings=128)
        model = transformers.LlamaForCausalLM(config)
    else:
        config = transformers.OlmoeConfig(vocab_size=97, hidden_size=24, intermediate_size=16,
                                           num_hidden_layers=2, num_attention_heads=4,
                                           num_key_value_heads=2, num_experts=4,
                                           num_experts_per_tok=2, max_position_embeddings=128,
                                           norm_topk_prob=False, eos_token_id=2, pad_token_id=1)
        model = transformers.OlmoeForCausalLM(config)
    model.to(torch.bfloat16).eval()
    model.save_pretrained(path, safe_serialization=True)
    if kind == "olmoe":
        # Recent transformers uses fused expert parameters in memory, while the published OLMoE
        # checkpoint has one gate/up/down tensor per expert. Write that same legacy layout.
        file = path / "model.safetensors"
        tensors = load_file(file)
        if "model.layers.0.mlp.experts.gate_up_proj" in tensors:
            for i in range(config.num_hidden_layers):
                prefix = f"model.layers.{i}.mlp.experts"
                fused = tensors.pop(prefix + ".gate_up_proj")
                down = tensors.pop(prefix + ".down_proj")
                for e in range(config.num_experts):
                    gate, up = fused[e].chunk(2, dim=0)
                    tensors[f"{prefix}.{e}.gate_proj.weight"] = gate.contiguous()
                    tensors[f"{prefix}.{e}.up_proj.weight"] = up.contiguous()
                    tensors[f"{prefix}.{e}.down_proj.weight"] = down[e].contiguous()
            save_file(tensors, str(file))
    return model.float().eval()


@pytest.mark.parametrize("kind", ["llama", "olmoe"])
def test_standard_decoder_matches_hf_float32_and_greedy(tmp_path, kind):
    reference = _make_model(tmp_path, kind)
    engine = StandardEngine.load(tmp_path, dtype=torch.float32)
    ids = torch.randint(0, 97, (2, 16), generator=torch.Generator().manual_seed(22))
    reference_layers = []
    hooks = [layer.register_forward_hook(lambda _m, _i, out: reference_layers.append(out[0] if isinstance(out, tuple) else out))
             for layer in reference.model.layers]
    with torch.inference_mode():
        expected = reference(ids, output_hidden_states=True, use_cache=False)
    for hook in hooks:
        hook.remove()
    actual_layers = []
    engine.model.forward(ids, capture_layers=actual_layers)
    assert len(actual_layers) == len(reference_layers) == reference.config.num_hidden_layers
    for actual, wanted in zip(actual_layers, reference_layers):
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(2), wanted.flatten(2), dim=-1)
        assert cosine.min() >= 0.9999
    trace = engine.trace(ids, decode_from=11)
    for actual, wanted in zip(trace.hidden_states, expected.hidden_states):
        cosine = torch.nn.functional.cosine_similarity(actual.flatten(2), wanted.flatten(2), dim=-1)
        assert cosine.min() >= 0.9999
    assert torch.allclose(trace.logits, expected.logits, atol=2e-5, rtol=2e-5)
    prompts = [row.tolist() for row in ids[:, :8]]
    actual = engine.generate(prompts, max_new_tokens=8, eos_token_ids=set()).tokens
    for prompt, tokens in zip(prompts, actual):
        sequence = torch.tensor([prompt])
        for token in tokens:
            with torch.inference_mode():
                next_id = int(reference(sequence, use_cache=False).logits[0, -1].argmax())
            assert token == next_id
            sequence = torch.cat((sequence, torch.tensor([[token]])), dim=1)


def test_standard_decoder_mixed_mode_b_c_export_and_run(tmp_path):
    source, target = tmp_path / "source", tmp_path / "mixed"
    _make_model(source, "llama")
    policy = Policy("mixed-standard", [Rule(match=r"self_attn\.q_proj$", mode="C", k=1,
                                               group=16, packing="tq1", act="bf16")],
                    Rule(match=".*", mode="B"))
    export(source, target, policy, shard_bytes=1 << 20, log=lambda *_: None)
    meta = json.loads((target / "manifest.json").read_text())
    assert any(entry["mode"] == "B" for entry in meta["tensors"].values())
    assert any(entry["mode"] == "C" for entry in meta["tensors"].values())
    engine = StandardEngine.load(target)
    result = engine.trace(torch.tensor([[3, 8, 2, 17]]), decode_from=3)
    assert result.logits.shape == (1, 4, 97)
    assert torch.isfinite(result.logits).all()


@pytest.mark.parametrize("kind", ["llama", "olmoe"])
def test_standard_decoder_mode_b_matches_source(tmp_path, kind):
    source, target = tmp_path / "source", tmp_path / "lossless"
    _make_model(source, kind)
    policy = Policy("lossless", default=Rule(match=".*", mode="B"))
    export(source, target, policy, shard_bytes=1 << 20, log=lambda *_: None)
    a, b = WeightStore(source), WeightStore(target)
    assert b.coded
    for name in a.names():
        assert torch.equal(a.get(name).contiguous().view(torch.uint8), b.get(name).contiguous().view(torch.uint8))
    ids = torch.tensor([[3, 8, 2, 17, 12]])
    original = StandardEngine.load(source).trace(ids, decode_from=3)
    coded = StandardEngine.load(target, mode_b_cache_bytes=0).trace(ids, decode_from=3)
    assert all(torch.equal(x.contiguous().view(torch.uint8), y.contiguous().view(torch.uint8))
               for x, y in zip(original.hidden_states, coded.hidden_states))
    assert torch.equal(original.logits.contiguous().view(torch.uint8), coded.logits.contiguous().view(torch.uint8))


def test_engine_dispatch_and_cli_benchmark(tmp_path, capsys):
    from boilerplate.cli import main

    _make_model(tmp_path, "llama")
    engine = Engine.load(tmp_path, max_seq_len=20, max_seqs=2, max_tokens=40)
    assert isinstance(engine, StandardEngine)
    assert engine.cfg.vocab_size == 97
    with pytest.raises(ValueError, match="cache limits"):
        engine.trace(torch.ones(3, 5, dtype=torch.long))
    with pytest.raises(ValueError, match="dequantization cache"):
        Engine.load(tmp_path, dequant_cache_bytes=1024)
    capsys.readouterr()
    threads = torch.get_num_threads()
    try:
        assert main(["bench", str(tmp_path), "--batch", "2", "--prompt-len", "4", "--gen-len", "2", "--threads", "2"]) == 0
        metrics = json.loads(capsys.readouterr().out)
        assert metrics["kernel_backend"] == "torch_standard"
        assert metrics["batch"] == 2 and metrics["peak_rss_bytes"] > 0
        assert metrics["prefill_tok_s"] > 0 and metrics["decode_tok_s"] > 0
        output = tmp_path / "standard-trace"
        assert main(["trace", str(tmp_path), "--seq-len", "4", "--batch", "2", "--decode-from", "3", "--out", str(output)]) == 0
        assert output.with_suffix(".safetensors").is_file()
    finally:
        torch.set_num_threads(threads)


def test_batched_generate_matches_one_prompt_at_a_time(tmp_path):
    _make_model(tmp_path, "llama")
    engine = StandardEngine.load(tmp_path)
    prompts = [[3, 8, 2, 17], [5, 6, 7, 9], [11, 12, 13]]
    batched = engine.generate(prompts, max_new_tokens=6, eos_token_ids=set()).tokens
    single = [engine.generate([p], max_new_tokens=6, eos_token_ids=set()).tokens[0] for p in prompts]
    assert batched == single
    eos = single[0][1]
    stopped = engine.generate(prompts, max_new_tokens=6, eos_token_ids={eos})
    assert stopped.tokens[0] == single[0][:single[0].index(eos) + 1] and stopped.finished[0]


def test_absmean_ptq_export_matches_qat_quantizer(tmp_path):
    from boilerplate.ternary.fit import absmean_ternary
    from boilerplate.ternary.pack import unpack

    source, target = tmp_path / "source", tmp_path / "absmean"
    _make_model(source, "llama")
    with pytest.raises(ValueError, match="absmean"):
        Rule(match=".", mode="C", k=2, fit="absmean")
    policy = Policy("absmean", [Rule(match=r"layers\.\d+\.(self_attn|mlp)\.\w+_proj$", mode="C", k=1,
                                     group=8, packing="tq1", fit="absmean")])
    assert Policy.from_dict(policy.to_dict()).rules[0].fit == "absmean"
    export(source, target, policy, log=lambda *_: None)
    meta = json.loads((target / "manifest.json").read_text())["tensors"]
    base = "model.layers.0.self_attn.q_proj"
    assert meta[base]["fit"] == "absmean" and meta[base]["mode"] == "C"
    w = WeightStore(source).get(base + ".weight")
    term = absmean_ternary(w.float(), 8)
    stored = WeightStore(target)
    assert torch.equal(unpack(stored.get(base + ".tq0.packed"), "tq1", w.shape[1]), term.trits)
    assert torch.equal(stored.get(base + ".tq0.scale"), term.scale)
    logits = StandardEngine.load(target).trace(torch.tensor([[3, 8, 2, 17]])).logits
    assert torch.isfinite(logits).all()


def test_standard_gptq_calibration_export_and_fallback(tmp_path):
    from boilerplate.format.calibrate import calibrate

    source, target = tmp_path / "source", tmp_path / "gptq"
    _make_model(source, "olmoe")
    policy = Policy("gptq", [Rule(match=r"layers\.\d+\.(self_attn|mlp\.experts\.\d+)\.\w+_proj$", mode="C", k=1,
                                  group=8, packing="tq1", gptq=True)])
    engine = StandardEngine.load(source)
    seqs = [list(range(3, 19)), list(range(20, 36))]
    hessians = calibrate(engine, seqs, policy, log=lambda *_: None)
    attn = "model.layers.0.self_attn"
    assert torch.equal(hessians[attn + ".q_proj"], hessians[attn + ".v_proj"])  # shared input
    assert hessians[attn + ".q_proj"].shape == (24, 24)
    # an absurd minimum drops every Hessian: export falls back to the plain fit and says so
    none = calibrate(engine, seqs, policy, min_tokens=10**6, log=lambda *_: None)
    assert none == {}
    export(source, target, policy, hessians=hessians, log=lambda *_: None)
    meta = json.loads((target / "manifest.json").read_text())["tensors"]
    assert meta[attn + ".q_proj"]["gptq"] is True
    for base, entry in meta.items():
        if entry.get("mode") == "C" and base not in hessians:
            assert entry["gptq"] is False and entry["gptq_fallback"] == "too few calibration tokens"
    fallback = tmp_path / "fallback"
    export(source, fallback, policy, hessians={}, log=lambda *_: None)
    assert all(e["gptq_fallback"] for e in json.loads((fallback / "manifest.json").read_text())["tensors"].values()
               if e.get("mode") == "C")
    assert torch.isfinite(StandardEngine.load(target).trace(torch.tensor([[3, 8, 2]])).logits).all()


def test_standard_routing_capture(tmp_path):
    from boilerplate.qat.routing import topk_overlap

    _make_model(tmp_path, "olmoe")
    engine = StandardEngine.load(tmp_path)
    routing = []
    engine.model.forward(torch.tensor([[3, 8, 2, 17]]), capture_routing=routing)
    assert len(routing) == 2 and routing[0].shape == (4, 2)
    assert topk_overlap(routing[0], routing[0]) == 1.0
