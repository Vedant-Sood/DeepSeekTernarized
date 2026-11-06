"""Write WikiText-2 (raw) test and train text, rows joined with blank lines (the text every perplexity in docs/reports was measured on).

    python scripts/eval/get_wikitext2.py work/data      # -> wikitext2_test.txt, wikitext2_train.txt
"""
import sys
from pathlib import Path

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download

out = Path(sys.argv[1])
out.mkdir(parents=True, exist_ok=True)
for split in ("test", "train"):
    f = hf_hub_download("Salesforce/wikitext", f"wikitext-2-raw-v1/{split}-00000-of-00001.parquet", repo_type="dataset")
    (out / f"wikitext2_{split}.txt").write_text("\n\n".join(pq.read_table(f).column("text").to_pylist()), encoding="utf-8")
    print(out / f"wikitext2_{split}.txt")
