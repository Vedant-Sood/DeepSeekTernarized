"""Cut a small, *trained* model out of a real V4.x checkpoint, fetching only the bytes it needs.

The result is a self-consistent checkpoint in the same on-disk format: a chosen subset of layers
(renumbered), the first ``n_experts`` routed experts of each (router rows sliced to match), the
real embeddings, head and tokenizer. It is not a good language model -- layers are missing -- but
every tensor holds trained values, so the engine-vs-reference comparison runs on realistic
activations, routing and attention patterns instead of random init.

Picking layers ``0,1,2,3,20,21,24`` of V4.1 Flash gives, after renumbering, two window-only layers,
a 2:1 KV-source layer with its own indexer and a layer sharing it, a 1:1 KV-source layer that is
also the candidate-block source, a layer sharing it, and an index-source layer that reads the
shared index keys inside the candidate blocks -- every attention path of the full model.

Tensors are read with HTTP range requests against the Hub (or from a local checkpoint directory),
so a 6 GB slice of the 510 GB release downloads in minutes. Engram tables (~100 GB each) cannot
be sliced by row without changing the hash layout, so engram layers are dropped by default.
"""

from __future__ import annotations

import concurrent.futures as cf
import copy
import json
import struct
import urllib.request
from pathlib import Path

import torch
from safetensors.torch import save_file

from ..engine.weights import ST_DTYPES

HUB = "https://huggingface.co/{repo}/resolve/{rev}/{file}"


def _remap(ids, keep: list[int]) -> list[int]:
    return [keep.index(i) for i in ids if i in keep]


def truncated_config(config: dict, layers: list[int], n_experts: int, keep_engram: bool = False) -> dict:
    cfg = copy.deepcopy(config)
    cfg.pop("vision_config", None)
    if "architectures" in cfg:
        cfg["architectures"] = ["DeepseekV41ForCausalLM"]
    t = cfg["text_config"] if "text_config" in cfg else cfg
    ratios = t.get("compress_ratios") or [0] * t["num_hidden_layers"]
    kv = _remap(t.get("kv_source_layer_ids") or [], layers)
    idx = _remap(t.get("index_source_layer_ids") or [], layers)
    new_ratios = [ratios[i] for i in layers]
    for j, r in enumerate(new_ratios):  # every compressed layer needs a same-ratio source at or before it
        srcs = [s for s in kv if s <= j]
        if r and (not srcs or new_ratios[max(srcs)] != r):
            raise ValueError(f"layer {layers[j]} reads compressed KV from a layer that was dropped")
    cand = t.get("candidate_source_layer_id")
    t.update(
        num_hidden_layers=len(layers),
        compress_ratios=new_ratios,
        kv_source_layer_ids=kv,
        index_source_layer_ids=idx,
        candidate_source_layer_id=layers.index(cand) if cand in layers else -1,
        n_routed_experts=n_experts,
        num_experts_per_tok=min(t["num_experts_per_tok"], n_experts),
        num_nextn_predict_layers=0,
        dspark_block_size=0,
        dspark_target_layer_ids=[],
    )
    if keep_engram:
        t["engram_layer_ids"] = _remap(t.get("engram_layer_ids") or [], layers)
    else:
        t["engram_layer_ids"] = []
    t["truncated_from"] = {"layers": layers, "n_experts": n_experts}
    return cfg


def plan(names: list[str], layers: list[int], n_experts: int, keep_engram: bool) -> dict[str, str]:
    """source tensor name -> new name, for every tensor the truncated model keeps."""
    out = {}
    for n in names:
        if n in ("embed.weight", "head.weight", "norm.weight"):
            out[n] = n
            continue
        p = n.split(".")
        if p[0] != "layers" or int(p[1]) not in layers:
            continue
        if ".engram." in n and not keep_engram:
            continue
        if ".experts." in n and int(p[4]) >= n_experts:
            continue
        if n.endswith("gate.bias_vl"):
            continue
        out[n] = ".".join(["layers", str(layers.index(int(p[1])))] + p[2:])
    return out


class _Reader:
    """Byte ranges of shard files, local or on the Hub."""

    def __init__(self, source: str, revision: str = "main"):
        self.source, self.revision = source, revision
        self.local = Path(source).exists()

    def read(self, file: str, start: int, end: int) -> bytes:
        if self.local:
            with open(Path(self.source) / file, "rb") as f:
                f.seek(start)
                return f.read(end - start)
        url = HUB.format(repo=self.source, rev=self.revision, file=file)
        for attempt in range(5):
            try:
                req = urllib.request.Request(url, headers={"Range": f"bytes={start}-{end - 1}"})
                with urllib.request.urlopen(req, timeout=120) as r:
                    data = r.read()
                if len(data) != end - start:
                    raise IOError(f"short read {len(data)} != {end - start}")
                return data
            except Exception:
                if attempt == 4:
                    raise
        raise AssertionError

    def text(self, file: str) -> str:
        if self.local:
            return (Path(self.source) / file).read_text()
        with urllib.request.urlopen(HUB.format(repo=self.source, rev=self.revision, file=file), timeout=120) as r:
            return r.read().decode()

    def header(self, file: str) -> tuple[int, dict]:
        (n,) = struct.unpack("<Q", self.read(file, 0, 8))
        return 8 + n, json.loads(self.read(file, 8, 8 + n))


def make_truncated(
    source: str,
    out_dir: str | Path,
    layers: list[int],
    n_experts: int,
    keep_engram: bool = False,
    revision: str = "main",
    shard_bytes: int = 2 << 30,
    workers: int = 8,
    log=print,
) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    rd = _Reader(source, revision)
    config = json.loads(rd.text("config.json"))
    weight_map = json.loads(rd.text("model.safetensors.index.json"))["weight_map"]
    keep = plan(list(weight_map), layers, n_experts, keep_engram)
    files = sorted({weight_map[n] for n in keep})
    headers = dict(zip(files, cf.ThreadPoolExecutor(workers).map(rd.header, files)))

    # merge nearby ranges per shard into requests of up to 256 MB
    jobs = []
    for f in files:
        start, meta = headers[f]
        items = sorted(((meta[n]["data_offsets"], n) for n in keep if weight_map[n] == f))
        cur: list = []
        for (b, e), n in items:
            if cur and (b - cur[-1][0][1] > (1 << 20) or e - cur[0][0][0] > (256 << 20)):
                jobs.append((f, cur))
                cur = []
            cur.append(((b, e), n))
        if cur:
            jobs.append((f, cur))
    offs = [headers[weight_map[n]][1][n]["data_offsets"] for n in keep]
    total = sum(e - b for b, e in offs)
    log(f"fetching {len(keep)} tensors, {total / 1e9:.2f} GB in {len(jobs)} requests")

    def fetch(job):
        f, items = job
        start, meta = headers[f]
        b0, e1 = items[0][0][0], items[-1][0][1]
        blob = rd.read(f, start + b0, start + e1)
        res = {}
        for (b, e), n in items:
            m = meta[n]
            t = torch.frombuffer(bytearray(blob[b - b0 : e - b0]), dtype=torch.uint8)
            if m["dtype"] == "F4":
                shape = list(m["shape"])
                shape[-1] //= 2
                res[n] = t.view(shape).view(torch.float4_e2m1fn_x2)
            else:
                res[n] = t.view(ST_DTYPES[m["dtype"]]).view(m["shape"])
        return res

    tensors: dict[str, torch.Tensor] = {}
    done = 0
    with cf.ThreadPoolExecutor(workers) as ex:
        for res in ex.map(fetch, jobs):
            for n, t in res.items():
                new = keep[n]
                if new.endswith("ffn.gate.weight") or new.endswith("ffn.gate.bias"):
                    t = t[:n_experts].clone()
                tensors[new] = t
                done += t.numel() * t.element_size()
            log(f"  {done / 1e9:.2f} GB")

    # shard and write
    weight_map_out, shard, size, k = {}, {}, 0, 0
    names = sorted(tensors)
    shards: list[dict] = []
    for n in names:
        nb = tensors[n].numel() * tensors[n].element_size()
        if shard and size + nb > shard_bytes:
            shards.append(shard)
            shard, size = {}, 0
        shard[n] = tensors[n]
        size += nb
    if shard:
        shards.append(shard)
    for k, sh in enumerate(shards):
        fname = f"model-{k + 1:05d}-of-{len(shards):05d}.safetensors"
        save_file({n: t.contiguous() for n, t in sh.items()}, str(out / fname))
        weight_map_out.update({n: fname for n in sh})
    total_out = sum(t.numel() * t.element_size() for t in tensors.values())
    (out / "model.safetensors.index.json").write_text(json.dumps({"metadata": {"total_size": total_out}, "weight_map": weight_map_out}, indent=2))
    (out / "config.json").write_text(json.dumps(truncated_config(config, layers, n_experts, keep_engram), indent=2))
    for f in ("tokenizer.json", "tokenizer_config.json"):
        try:
            (out / f).write_text(rd.text(f))
        except Exception:
            log(f"no {f}")
    return out
