"""Mode A engine vs the official DeepSeek reference (vendored in third_party/), on the tiny checkpoint.

The tiny checkpoint has every attention path of V4.1 (window-only layers, 2:1 and 1:1 compressed KV
shared across layers, index sources with and without their own keys, candidate-block prefilter),
engram lookups, hyper-connections and FP4 experts. Prompts are longer than the window and than
index_topk so the window ring wraps and the indexer has to choose.
"""

from pathlib import Path

import pytest
import torch

from boilerplate.engine import Engine, KVCache
from boilerplate.engine.model import _match_alignment
from boilerplate.engine.engram import EngramHasher, EngramLayout, compressed_token_map, is_prime
from boilerplate.reference import compare_traces
from boilerplate.reference.official_backend import OfficialReference, load_hf_tokenizer
from boilerplate.testing.tiny import tiny_tokenizer, write_tiny_checkpoint

REF_CODE = Path(__file__).resolve().parents[1] / "third_party" / "deepseek_v41_inference"


def test_hc_matrix_can_match_reference_alignment():
    """Identical mapped weights can start at different addresses in different exports."""
    generator = torch.Generator().manual_seed(42)
    reference = torch.randn(24, 512, generator=generator)
    x = torch.randn(1, 1, 512, generator=generator)
    storage = torch.empty(reference.numel() + 32, dtype=torch.float32)
    other_storage = torch.empty(reference.numel() + 32, dtype=torch.float32)
    for alignment in (0, 8, 24, 56):
        offset = next(i for i in range(16) if (storage.data_ptr() + 4 * i) % 64 == alignment)
        source = storage[offset:offset + reference.numel()].view_as(reference)
        source.copy_(reference)
        other_offset = next(i for i in range(16) if (other_storage.data_ptr() + 4 * i) % 64 == (alignment + 4) % 64)
        other = other_storage[other_offset:other_offset + reference.numel()].view_as(reference)
        other.copy_(reference)
        copied = _match_alignment(other, alignment)
        assert copied.data_ptr() % 64 == alignment
        assert torch.equal(copied, reference)
        assert torch.equal(torch.nn.functional.linear(x, source), torch.nn.functional.linear(x, copied))


def _need_reference_deps():
    for mod in ("sympy", "PIL", "transformers"):
        pytest.importorskip(mod)


@pytest.fixture(scope="module")
def tiny(tmp_path_factory):
    return write_tiny_checkpoint(tmp_path_factory.mktemp("tiny"))


@pytest.fixture(scope="module")
def engine(tiny):
    return Engine.load(tiny, max_seq_len=256)


def test_engine_aligns_hc_matrices_for_reference(engine):
    for layer in engine.model.layers:
        for fn, _, _ in layer.hc.values():
            assert fn.data_ptr() % 64 == 0


@pytest.fixture(scope="module")
def reference(tiny, tmp_path_factory):
    _need_reference_deps()
    return OfficialReference(
        REF_CODE, tiny, tmp_path_factory.mktemp("conv"), max_seq_len=256, max_batch_size=2, tokenizer=load_hf_tokenizer(tiny)
    )


def _ids(b, s, vocab=512, seed=0):
    return torch.randint(0, vocab, (b, s), generator=torch.Generator().manual_seed(seed))


def _assert_exact(ref, cand):
    cmp = compare_traces(ref, cand)
    assert all(m.bit_exact for m in cmp.layers), cmp.table()
    assert cmp.logits.bit_exact, cmp.table()


def test_prefill_bit_exact_vs_reference(reference, engine):
    ids = _ids(1, 40)
    _assert_exact(reference.trace(ids), engine.trace(ids))


def test_prefill_then_decode_bit_exact_vs_reference(reference, engine):
    """Batch of two, 13-token prefill then 27 decode steps: window ring wraps, compression groups
    complete mid-decode, candidate blocks and top-k change step to step."""
    ids = _ids(2, 40, seed=1)
    _assert_exact(reference.trace(ids, decode_from=13), engine.trace(ids, decode_from=13))


def test_short_prompts_shorter_than_ratio_and_window(reference, engine):
    ids = _ids(1, 12, seed=2)
    _assert_exact(reference.trace(ids, decode_from=1), engine.trace(ids, decode_from=1))


def test_mixed_batch_equals_separate_runs(engine):
    """Sequences of different lengths prefilled in one step, then decoded together, give the same
    tokens and near-identical logits as running each alone (only GEMM batch shapes differ)."""
    prompts = [_ids(1, 21, seed=3)[0].tolist(), _ids(1, 6, seed=4)[0].tolist(), _ids(1, 33, seed=5)[0].tolist()]
    together = engine.generate(prompts, max_new_tokens=12, temperature=0)
    alone = [engine.generate([p], max_new_tokens=12, temperature=0).tokens[0] for p in prompts]
    assert together.tokens == alone


def test_paged_cache_reuses_pages_without_leaks(engine):
    cache = engine.new_cache()
    a, b = cache.new_sequence(), cache.new_sequence()
    engine.model.forward([(a, list(range(1, 150))), (b, list(range(1, 10)))], cache)
    used = cache.pages_in_use
    assert used == -(-149 // cache.page_size) + 1
    cache.free(a)
    assert cache.pages_in_use == 1
    c = cache.new_sequence()  # reuses a's pages and ring slot, must not see a's state
    lg_c = engine.model.forward([(c, [5, 6, 7])], cache)
    fresh = engine.new_cache()
    lg_fresh = engine.model.forward([(fresh.new_sequence(), [5, 6, 7])], fresh)
    assert torch.equal(lg_c, lg_fresh)


def test_greedy_generation_matches_reference_argmax(reference, engine):
    """Greedy tokens from the engine are the reference's argmax when it is fed the same tokens."""
    prompt = _ids(1, 15, seed=6)[0].tolist()
    gen = engine.generate([prompt], max_new_tokens=10, temperature=0).tokens[0]
    ids = torch.tensor([prompt + gen])
    ref = reference.trace(ids, decode_from=len(prompt))
    ref_next = ref.logits[0, len(prompt) - 1 : -1].argmax(-1).tolist()
    assert ref_next == gen


# ---------------------------------------------------------------------------------- engram hashing


def test_is_prime():
    import sympy

    for n in list(range(-3, 2000)) + [16000057, 2**31 - 1, 2**61 - 1, 1_000_000_007 * 3]:
        assert is_prime(n) == bool(sympy.isprime(n)), n


def test_engram_layout_and_hashes_match_reference():
    """Primes, offsets, multipliers, compressed token map and hash ids equal the reference
    NgramHashState, for a prefill followed by decode steps."""
    _need_reference_deps()
    import sys

    sys.path.insert(0, str(REF_CODE))
    try:
        import engram as ref_engram
    finally:
        sys.path.remove(str(REF_CODE))
    from transformers import PreTrainedTokenizerFast

    from boilerplate.reference.official_backend import hf_config_to_model_args
    from boilerplate.testing.tiny import tiny_config

    cfg = tiny_config()
    t = cfg["text_config"]
    args = type("A", (), {**hf_config_to_model_args(cfg, text_only=True), "max_batch_size": 1, "max_seq_len": 64})()
    ref_layout = ref_engram.EngramLayout.from_args(args)
    ours = EngramLayout.from_config(t)
    assert ours.primes == ref_layout.primes
    assert ours.table_rows() == list(t["engram_num_embeddings"])

    tok = tiny_tokenizer(t["vocab_size"])
    hf_tok = PreTrainedTokenizerFast(tokenizer_object=tok)
    assert compressed_token_map(tok) == ref_engram.build_compressed_token_map(hf_tok)
    ref_state = ref_engram.NgramHashState(args, ref_layout, hf_tok)
    hasher = EngramHasher(t, ours, tok)
    ids = _ids(1, 20, seed=7)
    hist = torch.full((hasher.history_len,), -1, dtype=torch.long)
    got, hist = hasher(ids[0, :9], hist)
    assert torch.equal(got, ref_state(ids[:, :9], 0)[0])
    for p in range(9, 20):
        got, hist = hasher(ids[0, p : p + 1], hist)
        assert torch.equal(got, ref_state(ids[:, p : p + 1], p)[0])


def test_kv_cache_rejects_bad_page_size(engine):
    with pytest.raises(ValueError):
        KVCache(engine.cfg, page_size=63)
