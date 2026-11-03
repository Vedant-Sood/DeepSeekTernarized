"""Sample random row ranges of the real engram tables (weight FP8 + e8m0 scales) via HTTP range requests."""
import json, os, struct, random, urllib.request
import torch
from safetensors.torch import save_file
REPO = "https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017/resolve/main/"
tok = open(os.path.expanduser("~/.cache/huggingface/token")).read().strip() if os.path.exists(os.path.expanduser("~/.cache/huggingface/token")) else None
def get(url, a, b):
    req = urllib.request.Request(url, headers={"Range": f"bytes={a}-{b - 1}", **({"Authorization": f"Bearer {tok}"} if tok else {})})
    return urllib.request.urlopen(req, timeout=120).read()
h = json.load(open("/home/user/work/real_headers.json"))
random.seed(0)
ROWS, SLICES = 4096, 32
for layer in (1, 14):
    w = h[f"layers.{layer}.engram.embed.weight"]; s = h[f"layers.{layer}.engram.embed.scale"]
    url = REPO + w[2]
    (n,) = struct.unpack("<Q", get(url, 0, 8)); base = 8 + n
    N = w[1][0]
    ws, ss, rows = [], [], []
    for i in range(SLICES):
        r = random.randrange(0, N - ROWS)
        wb = get(url, base + w[3][0] + r * 256, base + w[3][0] + (r + ROWS) * 256)
        sb = get(url, base + s[3][0] + r * 8, base + s[3][0] + (r + ROWS) * 8)
        ws.append(torch.frombuffer(bytearray(wb), dtype=torch.uint8).view(ROWS, 256))
        ss.append(torch.frombuffer(bytearray(sb), dtype=torch.uint8).view(ROWS, 8))
        rows.append(r)
    save_file({"weight": torch.cat(ws).view(torch.float8_e4m3fn), "scale": torch.cat(ss).view(torch.float8_e8m0fnu), "rows": torch.tensor(rows)},
              f"/home/user/work/engram_sample/layer{layer}.safetensors")
    print(layer, "sampled", SLICES * ROWS, "rows of", N)
