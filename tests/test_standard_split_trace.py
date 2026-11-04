import hashlib
import json

import pytest
import torch
from safetensors.torch import load_file, save_file

from scripts.eval.standard_split_trace import main


def test_split_trace_resume_and_complete_comparison(tmp_path, capsys):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(84)
    config = transformers.LlamaConfig(vocab_size=37, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, max_position_embeddings=64)
    checkpoint = tmp_path / "model"
    transformers.LlamaForCausalLM(config).eval().save_pretrained(checkpoint, safe_serialization=True)
    ids = torch.randint(0, config.vocab_size, (2, 5), generator=torch.Generator().manual_seed(85))
    inputs = tmp_path / "ids.safetensors"
    save_file({"input_ids": ids}, str(inputs))
    ref, candidate = tmp_path / "ref", tmp_path / "candidate"
    previous_threads = torch.get_num_threads()
    try:
        def run(backend, out, *extra):
            return main(["record", backend, str(checkpoint), str(inputs), str(out), "--tokens", "4",
                         "--threads", "2", *extra])
        assert run("hf", ref, "--count", "1") == 0
        assert run("engine", candidate, "--count", "1") == 0
        assert main(["compare", str(ref), str(candidate), "--threads", "2"]) == 1
        assert main(["compare", str(ref), str(candidate), "--threads", "2", "--allow-partial"]) == 0
        # Simulate an interruption between the atomic trace rename and metadata update.
        info = json.loads((ref / "run.json").read_text())
        info["completed"].pop("0")
        (ref / "run.json").write_text(json.dumps(info))
        assert run("hf", ref, "--count", "1", "--resume") == 0
        assert json.loads((ref / "run.json").read_text())["completed"]["0"]["recovered_after_interruption"]
        assert run("hf", ref, "--start", "1", "--count", "1", "--resume") == 0
        assert run("engine", candidate, "--start", "1", "--count", "1", "--resume") == 0
        output = tmp_path / "comparison.json"
        assert main(["compare", str(ref), str(candidate), "--threads", "2", "--out", str(output)]) == 0
        result = json.loads(output.read_text())
        assert result["complete"] and result["prompts_compared"] == 2
        assert result["prompts_with_identical_greedy_tokens"] == 2
        assert min(result["layer_min_cosine_same_prefix"]) >= 0.9999
        path = candidate / "prompt-000.safetensors"
        tensors = {key: value.clone() for key, value in load_file(str(path)).items()}
        tensors["layer.0"][0, 0, 0] = float("nan")
        temporary = candidate / "replacement.safetensors"
        save_file(tensors, str(temporary))
        temporary.replace(path)
        info = json.loads((candidate / "run.json").read_text())
        info["completed"]["0"]["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        (candidate / "run.json").write_text(json.dumps(info))
        with pytest.raises(ValueError, match="non-finite cosine"):
            main(["compare", str(ref), str(candidate), "--threads", "2"])
    finally:
        torch.set_num_threads(previous_threads)
        capsys.readouterr()
