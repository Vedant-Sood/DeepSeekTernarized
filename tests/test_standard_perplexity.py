import json

import pytest
import torch
from safetensors.torch import save_file

from scripts.eval.standard_perplexity import main


def test_standard_perplexity_resumes_and_scores_llama_windows(tmp_path, capsys):
    transformers = pytest.importorskip("transformers")
    torch.manual_seed(53)
    config = transformers.LlamaConfig(vocab_size=31, hidden_size=16, intermediate_size=32,
                                      num_hidden_layers=2, num_attention_heads=4,
                                      num_key_value_heads=2, max_position_embeddings=64)
    model = transformers.LlamaForCausalLM(config).eval()
    checkpoint = tmp_path / "model"
    model.save_pretrained(checkpoint, safe_serialization=True)
    ids = torch.randint(0, config.vocab_size, (32,), generator=torch.Generator().manual_seed(54))
    tokens = tmp_path / "tokens.safetensors"
    save_file({"input_ids": ids}, str(tokens))
    resumed, uninterrupted = tmp_path / "resume.json", tmp_path / "full.json"
    previous_threads = torch.get_num_threads()
    try:
        base = [str(checkpoint), str(tokens), "--context", "8", "--threads", "2"]
        assert main(base + ["--state", str(resumed), "--max-chunks", "1"]) == 0
        partial = json.loads(resumed.read_text())
        assert partial["completed_chunks"] == 1 and not partial["complete"]
        assert main(base + ["--state", str(resumed), "--resume"]) == 0
        assert main(base + ["--state", str(uninterrupted)]) == 0
        a, b = json.loads(resumed.read_text()), json.loads(uninterrupted.read_text())
        assert a["complete"] and a["completed_chunks"] == 4 and a["scored_tokens"] == 12
        assert a["nll_sum"] == b["nll_sum"] and a["perplexity"] == b["perplexity"]
        expected = 0.0
        with torch.inference_mode():
            for chunk in ids.view(4, 8):
                logits = model(chunk.unsqueeze(0)).logits[:, 4:-1, :]
                labels = chunk[5:].unsqueeze(0)
                expected += float(torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels.flatten(), reduction="sum"))
        assert a["nll_sum"] == pytest.approx(expected, rel=1e-6, abs=1e-6)
    finally:
        torch.set_num_threads(previous_threads)
        capsys.readouterr()
