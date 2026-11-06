"""Routing capture in both engines and the routing-drift script."""

import json
import runpy
import sys

import pytest

from boilerplate.engine import Engine
from boilerplate.testing.tiny import write_tiny_checkpoint


def _run_script(args, capsys):
    argv = sys.argv
    sys.argv = ["routing_drift.py", *map(str, args)]
    try:
        with pytest.raises(SystemExit) as done:
            runpy.run_path("scripts/eval/routing_drift.py", run_name="__main__")
        assert done.value.code == 0
    finally:
        sys.argv = argv
    return json.loads(capsys.readouterr().out)


def test_v4_routing_capture_and_self_drift(tmp_path, capsys):
    ckpt = write_tiny_checkpoint(tmp_path / "tiny")
    eng = Engine.load(ckpt, max_seq_len=64)
    ids = list(range(5, 25))
    routing = eng.routing(ids)
    cfg = eng.cfg
    assert len(routing) == cfg.n_layers and all(r.shape == (len(ids), cfg.n_activated_experts) for r in routing)
    assert eng.model.routing_capture is None
    (tmp_path / "t.txt").write_text(" ".join(f"w{i % 30}" for i in range(400)))
    out = _run_script([ckpt, ckpt, tmp_path / "t.txt", "--ctx", "24", "--windows", "2"], capsys)
    assert out["mean_topk_overlap"] == 1.0 and out["windows"] == 2


def test_standard_routing_capture(tmp_path):
    from tests.test_standard_engine import _make_model

    _make_model(tmp_path, "olmoe")
    eng = Engine.load(tmp_path, max_seq_len=32)
    routing = eng.routing([3, 8, 2, 17])
    assert len(routing) == 2 and routing[0].shape == (4, 2)
