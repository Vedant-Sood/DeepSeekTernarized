"""Mode B on every tensor of a checkpoint: coded size per role, exactness, encode/decode throughput."""
import json, re, sys, time, torch
from boilerplate.engine.weights import WeightStore
from boilerplate.format import entropy
torch.set_num_threads(4)
ws = WeightStore(sys.argv[1])
def role(n):
    if ".experts." in n: return "routed_experts"
    if "engram.embed" in n: return "engram_table"
    if ".shared_experts." in n: return "shared_experts"
    if ".attn." in n: return "attention"
    if n.startswith("embed") or n.startswith("head"): return "embed_head"
    return "other"
agg, enc_t, dec_t, nbytes = {}, 0.0, 0.0, 0
for n in ws.names():
    t = ws.get(n)
    es = t.element_size()
    key = (role(n), "scale" if n.endswith(".scale") else str(t.dtype).replace("torch.", ""))
    a = agg.setdefault(key, [0, 0])
    a[0] += t.numel() * es
    if es > 2 or t.numel() < 4096:
        a[1] += t.numel() * es
        continue
    t0 = time.perf_counter(); p = entropy.encode(t); enc_t += time.perf_counter() - t0
    sym = entropy.split_bytes(t)[0]
    t0 = time.perf_counter(); back = entropy.decode(p, sym.numel()); dec_t += time.perf_counter() - t0
    assert torch.equal(back, t.contiguous().view(torch.uint8).flatten()), n
    nbytes += t.numel() * es
    a[1] += entropy.coded_bytes(p)
rows = {f"{r}/{d}": {"bytes": a[0], "coded": a[1], "ratio": a[1] / a[0]} for (r, d), a in sorted(agg.items())}
tot = [sum(a[0] for a in agg.values()), sum(a[1] for a in agg.values())]
print(json.dumps({"by_role_dtype": rows, "total": {"bytes": tot[0], "coded": tot[1], "ratio": tot[1] / tot[0]},
                  "all_coded_tensors_roundtrip_exact": True, "encode_MBps": nbytes / enc_t / 1e6, "decode_MBps": nbytes / dec_t / 1e6,
                  "threads": torch.get_num_threads()}, indent=1))
