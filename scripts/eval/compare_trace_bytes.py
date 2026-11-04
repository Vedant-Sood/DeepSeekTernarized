"""Check every saved hidden state and logit tensor for byte-identical values."""

import argparse
import json

from boilerplate.reference import ReferenceTrace
from modeb_validate import same_bits


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference")
    parser.add_argument("candidate")
    args = parser.parse_args()
    ref, cand = ReferenceTrace.load(args.reference), ReferenceTrace.load(args.candidate)
    if len(ref.hidden_states) != len(cand.hidden_states):
        raise ValueError("trace layer counts differ")
    ids_equal = same_bits(ref.input_ids, cand.input_ids)
    layers = [same_bits(a, b) for a, b in zip(ref.hidden_states, cand.hidden_states)]
    logits_equal = ref.logits is not None and cand.logits is not None and same_bits(ref.logits, cand.logits)
    print(json.dumps({"input_bytes_identical": ids_equal, "layer_bytes_identical": layers,
                      "logits_bytes_identical": logits_equal}, indent=2))
    return 0 if ids_equal and all(layers) and logits_equal else 1


if __name__ == "__main__":
    raise SystemExit(main())
