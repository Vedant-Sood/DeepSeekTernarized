"""GPTQ vs plain residual fit on real expert inputs of the 7-layer slice.

Calibration H from WikiText-2 train, evaluation H from WikiText-2 test (held out). Output error of a
fitted weight Q on inputs X is ||X (W - Q)^T||^2 / tokens = tr((W-Q) H (W-Q)^T) with H = X^T X / tokens.
"""
import json, sys, time, torch
from boilerplate.engine import Engine
from boilerplate.format.calibrate import calibrate
from boilerplate.format.export import dequant_source
from boilerplate.format.policy import Policy
from boilerplate.ternary.fit import fit_gptq, fit_residual, dequant_terms
torch.set_num_threads(4)
layer, nexp = int(sys.argv[1]), int(sys.argv[2])
pol = Policy.from_dict({"name": "g", "rules": [{"match": rf"^layers\.{layer}\.ffn\.experts\.([0-9]|1[0-5])\.w[123]$", "mode": "C", "k": 2, "packing": "tq1", "gptq": True}]})
eng = Engine.load("/home/user/work/trunc7", max_seq_len=520)
def seqs(path, n):
    t = eng.tokenizer.encode(open(path).read()[:2_000_000]).ids
    return [t[i:i + 512] for i in range(0, 512 * n, 512)]
t0 = time.time()
Htr = calibrate(eng, seqs("/home/user/work/data/wiki_train.txt", 8), pol, {layer}, log=lambda *a: None)
Hte = calibrate(eng, seqs("/home/user/work/data/wiki_test.txt", 4), pol, {layer}, log=lambda *a: None)
print("calibration", round(time.time() - t0), "s", flush=True)
ws = eng.model.w
out = {}
for b in sorted(Htr)[: 3 * nexp]:
    W = dequant_source(ws.get(f"{b}.weight"), ws.get(f"{b}.scale"))
    H = Hte[b]
    def oerr(Q):
        D = W - Q
        return (D @ H * D).sum().item()
    ref = (W @ H * W).sum().item()
    row = {}
    for k in (1, 2):
        row[f"k{k}_plain"] = oerr(dequant_terms(fit_residual(W, k, 128), W.size(1), 128)) / ref
        row[f"k{k}_gptq"] = oerr(dequant_terms(fit_gptq(W, Htr[b], k, 128), W.size(1), 128)) / ref
    out[b] = row
    print(b, {k: round(v, 4) for k, v in row.items()}, flush=True)
agg = {k: sum(r[k] for r in out.values()) / len(out) for k in next(iter(out.values()))}
print(json.dumps({"layer": layer, "weights": len(out), "mean_relative_output_error": agg}, indent=1))
