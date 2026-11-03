"""Full-model sizes from the real shard headers under each policy (analytic: bytes = params x bits)."""
import json, re, sys
h = json.load(open("/home/user/work/real_headers.json"))
gen = {k: v for k, v in h.items() if not k.startswith("mtp.") and "vision" not in k and not k.startswith("visual")}
def nbytes(v): return v[3][1] - v[3][0]
def numel(v):
    n = 1
    for d in v[1]: n *= d
    return n
roles = {}
for k, v in h.items():
    if k.endswith(".scale"): continue
    base = k[:-7] if k.endswith(".weight") else k
    r = "routed_experts" if re.search(r"\.ffn\.experts\.\d+\.w[123]\.weight$", k) else "engram_table" if k.endswith("engram.embed.weight") else "mtp_vision" if (k.startswith("mtp.") or "vision" in k or k.startswith("visual")) else "other"
    b = nbytes(v) + (nbytes(h[base + ".scale"]) if base + ".scale" in h else 0)
    p = numel(v) * (2 if v[0] == "F4" or (r == "routed_experts" and v[0] in ("I8", "U8")) else 1)
    a = roles.setdefault(r, [0, 0]); a[0] += p; a[1] += b
print(json.dumps({r: {"params": a[0], "GB": a[1] / 1e9, "bits": 8 * a[1] / a[0]} for r, a in roles.items()}, indent=1))
