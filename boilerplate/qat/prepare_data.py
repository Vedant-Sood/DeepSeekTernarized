"""Build a QAT training corpus and a held-out split from Hugging Face datasets, streaming.

    boilerplate qat-data work/data/smollm2 \
        --source HuggingFaceFW/fineweb-edu:sample/10BT/*.parquet:1.0 --train-chars 2.2e9 --heldout-chars 2e7

Each ``--source`` is ``repo_id:glob:weight`` (dataset repos), or a local directory or file with a
glob. Matching files are read in sorted order with range requests, so only the bytes needed are
downloaded: Parquet row groups through pyarrow, ``.json(l)`` optionally ``.gz``/``.zst``
compressed. A source's share of the character budget is its weight over the total.

**Held out.** The held-out documents come from each source's *last* matching file and the
training documents from its first files onward. Every shard-local document belongs to exactly one
split, so the two cannot overlap (unless a source has a single file, which is refused).

Outputs ``train.jsonl`` and ``heldout.jsonl`` (one ``{"text": ...}`` per line, for
``qat-tokenize``, which puts EOS between documents) and ``data.json`` recording every file read,
document and character counts, and the arguments. About 4 characters make one token for
English web text, so a 0.5 B-token run needs ~2.2e9 characters of train text (with margin).
"""

from __future__ import annotations

import fnmatch
import gzip
import io
import json
from pathlib import Path
from typing import Iterator


def _open(path: str, local: bool):
    if local:
        return open(path, "rb")
    from huggingface_hub import HfFileSystem

    return HfFileSystem().open(path, "rb", block_size=8 << 20)


def iter_documents(path: str, local: bool = False) -> Iterator[str]:
    """Texts of one dataset file, streamed."""
    if path.endswith(".parquet"):
        import pyarrow.parquet as pq

        with _open(path, local) as f:
            for batch in pq.ParquetFile(f).iter_batches(columns=["text"], batch_size=1024):
                for t in batch.column(0).to_pylist():
                    if t:
                        yield t
        return
    with _open(path, local) as raw:
        if path.endswith(".gz"):
            stream = gzip.GzipFile(fileobj=raw)
        elif path.endswith(".zst"):
            import zstandard  # optional: pip install zstandard

            stream = zstandard.ZstdDecompressor().stream_reader(raw)
        else:
            stream = raw
        for line in io.TextIOWrapper(stream, encoding="utf-8"):
            if line.strip():
                t = json.loads(line).get("text")
                if t:
                    yield t


def list_files(source: str) -> tuple[list[str], bool]:
    """Sorted files of ``repo_or_dir:glob`` (local when the part before the colon exists on disk)."""
    where, pattern = source.rsplit(":", 1)
    if Path(where).exists():
        return sorted(str(p) for p in Path(where).glob(pattern)), True
    from huggingface_hub import HfApi

    files = HfApi().list_repo_files(where, repo_type="dataset")
    return [f"datasets/{where}/{f}" for f in sorted(files) if fnmatch.fnmatch(f, pattern)], False


def prepare(out_dir: str | Path, sources: list[str], train_chars: float, heldout_chars: float,
            log=print) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    parsed = []
    for s in sources:
        spec, weight = s.rsplit(":", 1)
        files, local = list_files(spec)
        if len(files) < 2:
            raise ValueError(f"{spec}: need at least two files (the last one is held out), found {len(files)}")
        parsed.append((spec, float(weight), files, local))
    total_w = sum(w for _, w, _, _ in parsed)
    record = {"sources": [], "train_chars": 0, "heldout_chars": 0, "train_docs": 0, "heldout_docs": 0,
              "args": {"sources": sources, "train_chars": train_chars, "heldout_chars": heldout_chars}}
    with (out / "train.jsonl").open("w", encoding="utf-8") as train, \
            (out / "heldout.jsonl").open("w", encoding="utf-8") as held:
        for spec, w, files, local in parsed:
            share = w / total_w
            src = {"source": spec, "weight": w, "train_files": [], "heldout_file": files[-1]}
            for name, budget, sink, pool in (("heldout", heldout_chars * share, held, files[-1:]),
                                             ("train", train_chars * share, train, files[:-1])):
                chars = docs = 0
                for path in pool:
                    if chars >= budget:
                        break
                    if name == "train":
                        src["train_files"].append(path)
                    for text in iter_documents(path, local):
                        sink.write(json.dumps({"text": text}, ensure_ascii=False) + "\n")
                        chars += len(text)
                        docs += 1
                        if chars >= budget:
                            break
                if chars < budget:
                    log(f"warning: {spec} {name}: only {chars:,} of {int(budget):,} characters available")
                src[f"{name}_chars"], src[f"{name}_docs"] = chars, docs
                record[f"{name}_chars"] += chars
                record[f"{name}_docs"] += docs
                log(f"{spec} {name}: {docs:,} documents, {chars:,} characters")
            record["sources"].append(src)
    (out / "data.json").write_text(json.dumps(record, indent=2))
    return out


def main(a) -> int:
    prepare(a.out_dir, a.source, a.train_chars, a.heldout_chars)
    return 0
