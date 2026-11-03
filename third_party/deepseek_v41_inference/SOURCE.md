# Vendored: official DeepSeek V4.1 Flash reference inference code

Copied unmodified from `inference/` of the Hugging Face repo
[`deepseek-ai/DeepSeek-V4.1-Flash-1017`](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash-1017),
revision `1bae0e1088993b9b82de08f59de5be5a34e42170`, downloaded 2026-11-03. MIT licensed (see `LICENSE`).

It is the trusted reference that Boilerplate's engine is validated against: no mainstream runtime
supports `deepseek_v41`. It is vendored so the equivalence tests run without network access.

`kernel.py` needs TileLang and a GPU; on CPU, `boilerplate.reference.official_backend` replaces it with
`boilerplate.kernels.torch_ref`. `generate.py`, `run.sh` and the examples are not included.
