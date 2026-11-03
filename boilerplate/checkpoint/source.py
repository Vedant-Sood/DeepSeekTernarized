"""Locate a checkpoint's metadata: config.json, the safetensors index and (optionally) shard headers.

Works on a local directory or a Hugging Face repo id. For a repo id only metadata is fetched:
config.json and the index are small files, and shard headers are read with HTTP range requests.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .safetensors_io import TensorInfo, read_header, read_remote_headers


@dataclass
class CheckpointSource:
    location: str
    config: dict
    weight_map: dict[str, str]  # tensor name -> shard file
    headers: dict[str, TensorInfo] | None = None  # None when headers were not read
    file_sizes: dict[str, int] = field(default_factory=dict)  # shard -> bytes (if known)
    is_local: bool = True

    @property
    def tensor_names(self) -> list[str]:
        return list(self.weight_map)

    @classmethod
    def open(cls, location: str | Path, read_headers: bool = True, revision: str | None = None) -> "CheckpointSource":
        p = Path(location)
        if p.exists():
            return cls._open_local(p, read_headers)
        return cls._open_hub(str(location), read_headers, revision)

    @classmethod
    def _open_local(cls, root: Path, read_headers: bool) -> "CheckpointSource":
        config = json.loads((root / "config.json").read_text())
        index_path = root / "model.safetensors.index.json"
        shards = sorted(root.glob("*.safetensors"))
        if index_path.exists():
            weight_map = json.loads(index_path.read_text())["weight_map"]
        else:
            # single-file checkpoint: build the map from headers
            weight_map = {}
            for s in shards:
                for name in read_header(s):
                    weight_map[name] = s.name
        headers = None
        if read_headers:
            headers = {}
            for s in sorted(set(weight_map.values())):
                headers.update(read_header(root / s))
        sizes = {s.name: s.stat().st_size for s in shards}
        return cls(str(root), config, weight_map, headers, sizes, True)

    @classmethod
    def _open_hub(cls, repo_id: str, read_headers: bool, revision: str | None) -> "CheckpointSource":
        from huggingface_hub import HfApi, hf_hub_download

        config = json.loads(Path(hf_hub_download(repo_id, "config.json", revision=revision)).read_text())
        index = json.loads(
            Path(hf_hub_download(repo_id, "model.safetensors.index.json", revision=revision)).read_text()
        )
        headers = read_remote_headers(repo_id, revision) if read_headers else None
        info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
        sizes = {s.rfilename: s.size for s in info.siblings if s.rfilename.endswith(".safetensors") and s.size}
        return cls(repo_id, config, index["weight_map"], headers, sizes, False)
