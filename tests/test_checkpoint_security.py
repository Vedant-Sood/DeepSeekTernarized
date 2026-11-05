"""Adversarial checkpoint metadata must fail before reading outside a root or mapping data."""

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from boilerplate.checkpoint.paths import checkpoint_path
from boilerplate.checkpoint.safetensors_io import MAX_HEADER_BYTES, parse_header_bytes, read_header
from boilerplate.checkpoint.source import CheckpointSource
from boilerplate.engine.weights import WeightStore, _Shard
from boilerplate.testing.truncate import _Reader


@pytest.mark.parametrize("name", ["../outside.safetensors", "a/../../outside", "/tmp/outside",
                                   r"C:\outside", "C:outside", r"\\server\share",
                                   r"..\outside", "a//b", "a/./b", ""])
def test_index_paths_cannot_escape_checkpoint(tmp_path, name):
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": name}}))
    for load in (WeightStore, CheckpointSource.open,
                 lambda p: CheckpointSource.open(p, read_headers=False)):
        with pytest.raises(ValueError, match="checkpoint path"):
            load(tmp_path)
    with pytest.raises(ValueError, match="checkpoint path"):
        _Reader(str(tmp_path)).read(name, 0, 1)


def test_nested_shard_paths_are_supported(tmp_path):
    (tmp_path / "nested").mkdir()
    save_file({"x": torch.ones(2)}, str(tmp_path / "nested" / "weights.safetensors"))
    (tmp_path / "config.json").write_text("{}")
    (tmp_path / "model.safetensors.index.json").write_text(json.dumps({"weight_map": {"x": "nested/weights.safetensors"}}))
    assert torch.equal(WeightStore(tmp_path).get("x"), torch.ones(2))
    assert CheckpointSource.open(tmp_path).headers["x"].shape == (2,)


def test_external_symlinks_are_rejected(tmp_path):
    root = tmp_path / "checkpoint"
    root.mkdir()
    (tmp_path / "outside").write_bytes(b"x")
    try:
        (root / "linked").symlink_to(tmp_path / "outside")
    except OSError:
        pytest.skip("symlinks unavailable")
    with pytest.raises(ValueError, match="escapes"):
        checkpoint_path(root, "linked")


@pytest.mark.parametrize("size", [0, MAX_HEADER_BYTES + 1, 2**64 - 1, 32])
def test_invalid_header_lengths_fail_before_reading_or_mapping(tmp_path, size):
    path = tmp_path / "weights.safetensors"
    path.write_bytes(struct.pack("<Q", size) + b"{}")
    for load in (read_header, _Shard):
        with pytest.raises(ValueError, match="header length"):
            load(path)


@pytest.mark.parametrize("meta,data", [
    ({"dtype": "U8", "shape": [-1], "data_offsets": [0, 1]}, b"x"),
    ({"dtype": "U8", "shape": [True], "data_offsets": [0, 1]}, b"x"),
    ({"dtype": "U8", "shape": [1], "data_offsets": [-1, 0]}, b"x"),
    ({"dtype": "U8", "shape": [1], "data_offsets": [1, 0]}, b"x"),
    ({"dtype": "F32", "shape": [1], "data_offsets": [0, 1]}, b"x"),
    ({"dtype": "U8", "shape": [2], "data_offsets": [0, 2]}, b"x"),
    ({"dtype": "U8", "shape": [1], "data_offsets": [0, 1]}, b"xx"),
    ({"dtype": "F4", "shape": [2, 1], "data_offsets": [0, 1]}, b"x"),
    ({"dtype": "invalid", "shape": [1], "data_offsets": [0, 1]}, b"x"),
])
def test_malformed_tensor_headers_fail_in_both_readers(tmp_path, meta, data):
    path = tmp_path / "weights.safetensors"
    raw = json.dumps({"x": meta}).encode()
    path.write_bytes(struct.pack("<Q", len(raw)) + raw + data)
    for load in (read_header, _Shard):
        with pytest.raises(ValueError):
            load(path)


@pytest.mark.parametrize("ranges", [[(0, 1), (0, 1)], [(1, 2)], [(0, 1), (2, 3)]])
def test_header_gaps_and_overlaps_are_rejected(ranges):
    header = {str(i): {"dtype": "U8", "shape": [end-start], "data_offsets": [start, end]}
              for i, (start, end) in enumerate(ranges)}
    with pytest.raises(ValueError, match="gap or overlap"):
        parse_header_bytes(json.dumps(header).encode())


def test_duplicate_header_keys_are_rejected():
    with pytest.raises(ValueError, match="duplicate"):
        parse_header_bytes(b'{"x": {}, "x": {}}')


def test_empty_scalar_and_packed_tensors_still_load(tmp_path):
    values = {"scalar": torch.tensor(1.0), "empty": torch.empty(0),
              "packed": torch.zeros(2, 3, dtype=torch.uint8).view(torch.float4_e2m1fn_x2)}
    save_file(values, str(tmp_path / "weights.safetensors"))
    store = WeightStore(tmp_path)
    for name, value in values.items():
        assert store.get(name).shape == value.shape
        assert torch.equal(store.get(name).reshape(-1).view(torch.uint8), value.reshape(-1).view(torch.uint8))


def test_remote_header_length_is_bounded_before_second_request(monkeypatch):
    reader = _Reader("owner/model")
    calls = []
    def read(file, start, end):
        calls.append((start, end))
        return struct.pack("<Q", 2**64 - 1)
    monkeypatch.setattr(reader, "read", read)
    with pytest.raises(ValueError, match="header length"):
        reader.header("model.safetensors")
    assert calls == [(0, 8)]


def test_remote_filenames_and_revision_are_url_encoded():
    reader = _Reader("owner/model", revision="refs/pr/1")
    assert reader._url("a file.safetensors").endswith("/refs%2Fpr%2F1/a%20file.safetensors")
    with pytest.raises(ValueError, match="checkpoint path"):
        reader._url("../config.json")


def test_metadata_downloads_are_bounded(monkeypatch):
    import boilerplate.testing.truncate as truncate
    class Response:
        def __enter__(self):
            return self
        def __exit__(self, *args):
            return False
        def read(self, size):
            assert size == 5
            return b"12345"
    monkeypatch.setattr(truncate, "MAX_HEADER_BYTES", 4)
    monkeypatch.setattr(truncate.urllib.request, "urlopen", lambda *args, **kwargs: Response())
    with pytest.raises(ValueError, match="size limit"):
        _Reader("owner/model").text("config.json")
