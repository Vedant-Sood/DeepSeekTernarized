import json, torch
from safetensors.torch import load_file
from boilerplate.checkpoint.dequant import dequant_fp8_block
from boilerplate.ternary.fit import fit_residual, dequant_terms
from boilerplate.format import entropy
torch.set_num_threads(2)
res = {}
for L in (1, 14):
    d = load_file(f"/home/user/work/engram_sample/layer{L}.safetensors")
    w, s = d["weight"], d["scale"]
    W = w.float() * s.float().repeat_interleave(32, -1)
    r = {"rows": W.size(0)}
    for k, g in ((1, 128), (2, 128), (2, 256), (3, 128)):
        R = dequant_terms(fit_residual(W, k, g), 256, g)
        cos = torch.nn.functional.cosine_similarity(W, R, -1)
        r[f"k{k}_g{g}"] = {"rel_err": ((W - R).norm() / W.norm()).item(), "row_cos_mean": cos.mean().item(), "row_cos_p01": cos.quantile(0.01).item(), "bits": k * (1.6 + 16 / g)}
    # FP8 itself vs the BF16 value it stands for is exact; for reference, what does FP8->FP4-like k1 vs k2 mean: compare
    for name, t in (("weight_fp8", w), ("scale_e8m0", s)):
        p = entropy.encode(t)
        r[f"modeB_{name}"] = {"bits_per_byte": 8 * entropy.coded_bytes(p) / t.numel(), "entropy_bits": entropy.entropy_bits(t.view(torch.uint8))}
    res[L] = r
print(json.dumps(res, indent=1))
