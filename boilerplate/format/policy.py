"""Mixed-precision policy: per-tensor regex rules choosing a storage mode (YAML).

Mode C must actually shrink what it replaces: the exporter compares each rule's effective bits per
weight with the source tensor's (V4.1's routed experts are already 4.25-bit FP4) and refuses a
setting that is not below ``MAX_RATIO`` of it unless the rule sets ``allow_larger``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import yaml

MODES = ("A", "B", "C")
MAX_RATIO = 0.85


@dataclass
class Rule:
    match: str
    mode: str = "A"
    k: int = 2
    group: int = 128
    packing: str = "tq2"
    scale_dtype: str = "fp16"
    act: str = "int8"
    gptq: bool = False
    allow_larger: bool = False  # permit Mode C settings that do not shrink the source (experiments)

    def effective_bits(self) -> float:
        """Bits per weight of a Mode C setting: k terms of packed trits plus one 16-bit scale per group."""
        return self.k * ({"tq2": 2.0, "tq1": 1.6}[self.packing] + 16.0 / self.group)

    def __post_init__(self):
        self.regex = re.compile(self.match)
        if self.mode not in MODES:
            raise ValueError(f"mode {self.mode!r}: expected one of {MODES}")
        if self.mode == "C":
            if not 1 <= self.k <= 4:
                raise ValueError("k must be 1..4")
            if self.packing not in ("tq2", "tq1"):
                raise ValueError(f"packing {self.packing!r}")
            if self.scale_dtype not in ("fp16", "bf16"):
                raise ValueError(f"scale_dtype {self.scale_dtype!r}")
            if self.act not in ("int8", "bf16"):
                raise ValueError(f"act {self.act!r}")


@dataclass
class Policy:
    name: str
    rules: list[Rule] = field(default_factory=list)
    default: Rule = field(default_factory=lambda: Rule(match=".*"))

    def rule_for(self, base: str) -> Rule:
        for r in self.rules:
            if r.regex.search(base):
                return r
        return self.default

    @classmethod
    def from_dict(cls, d: dict) -> "Policy":
        default = Rule(match=".*", **(d.get("default") or {}))
        return cls(d.get("name", "policy"), [Rule(**r) for r in d.get("rules") or []], default)

    @classmethod
    def load(cls, path: str | Path) -> "Policy":
        return cls.from_dict(yaml.safe_load(Path(path).read_text()))

    def to_dict(self) -> dict:
        keep = ("match", "mode", "k", "group", "packing", "scale_dtype", "act", "gptq", "allow_larger")
        return {"name": self.name, "rules": [{k: getattr(r, k) for k in keep} for r in self.rules]}
