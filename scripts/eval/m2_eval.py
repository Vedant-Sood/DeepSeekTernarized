"""Compare Boilerplate exports against Mode A on the same tokens: per-layer cosine, logit KL/top-1, NLL."""
import json, sys, time
import torch, torch.nn.functional as F
from safetensors.torch import load_file
from boilerplate.engine import Engine
from boilerplate.reference import compare_traces
torch.set_num_threads(4)
base, cands, ids_path = sys.argv[1], sys.argv[2].split(","), sys.argv[3]
n = int(sys.argv[4]) if len(sys.argv) > 4 else None
ids = load_file(ids_path)["input_ids"] if ids_path.endswith(".safetensors") else torch.randint(0, int(ids_path), (1, n or 64))
if n: ids = ids[:, :n]
def nll(lg, ids):
    return F.cross_entropy(lg[0, :-1].float(), ids[0, 1:]).item()
def run(path):
    e = Engine.load(path, max_seq_len=ids.size(1) + 8)
    t = time.time(); tr = e.trace(ids); return tr, time.time() - t
ref, ta = run(base)
out = {"base": {"path": base, "nll": nll(ref.logits, ids), "ppl": float(torch.exp(torch.tensor(nll(ref.logits, ids)))), "sec": ta}}
for c in cands:
    tr, tc = run(c)
    cmp = compare_traces(ref, tr)
    m = json.loads(open(f"{c}/manifest.json").read())["summary"]
    l = nll(tr.logits, ids)
    out[c] = {"min_layer_cos_mean": min(x.cos_mean for x in cmp.layers), "final_cos_mean": cmp.layers[-1].cos_mean,
              "kl_mean": cmp.logits.kl_mean, "top1": cmp.logits.top1_agreement, "exact": cmp.logits.bit_exact,
              "nll": l, "ppl": float(torch.exp(torch.tensor(l))), "sec": tc, "bpw_total": m["bits_per_weight"],
              "by_mode": {k: round(v["bits_per_weight"], 3) for k, v in m["by_mode"].items()}}
    print(c, "\n" + cmp.table(), flush=True)
print(json.dumps(out, indent=1))
