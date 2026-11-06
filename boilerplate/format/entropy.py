"""Mode B: lossless entropy coding of weight bytes (canonical Huffman, DFloat11-style).

What gets coded depends on the element size:

* 1-byte formats (FP8 e4m3 weights, packed FP4 experts, e8m0 scales): every byte is a symbol.
  Coding the whole byte captures the exponent redundancy *and* any joint structure (an FP4 byte
  holds two weights).
* 2-byte formats (BF16/FP16): the high byte (sign, exponent, top mantissa bit for BF16) is coded,
  the low byte (mantissa, close to uniform) is stored raw.
* 4-byte formats are left alone (they are tiny in these checkpoints).

Stream layout (documented for the GPU decoder): symbols are split into blocks of ``block``
symbols; codes are written MSB-first back to back; ``offsets[b]`` is the bit where block b starts
(``offsets[-1]`` is the total). Each block decodes independently with one ``2^maxlen``-entry lookup
table, ``(symbol << 8) | code_length``, indexed by the next ``maxlen`` bits. Codes are canonical, so
the 256 code lengths are all a decoder needs.
"""

from __future__ import annotations

import heapq
import math

import torch

from ..kernels.cpu import ext

MAXLEN = 12
BLOCK = 4096


def _parameters(maxlen: int, block: int = BLOCK) -> None:
    if type(maxlen) is not int or not 1 <= maxlen <= MAXLEN:
        raise ValueError(f"maxlen must be in 1..{MAXLEN}")
    if type(block) is not int or block <= 0:
        raise ValueError("block must be positive")


def code_lengths(freq: torch.Tensor, maxlen: int = MAXLEN) -> torch.Tensor:
    """Huffman code lengths for 256 symbol counts, limited to maxlen bits."""
    _parameters(maxlen)
    if freq.shape != (256,) or freq.is_floating_point() or torch.any(freq < 0):
        raise ValueError("expected 256 nonnegative integer symbol counts")
    f = [int(v) for v in freq.tolist()]
    syms = [s for s in range(256) if f[s] > 0]
    if len(syms) > 1 << maxlen:
        raise ValueError("maxlen cannot represent all used symbols")
    lens = [0] * 256
    if len(syms) == 1:
        lens[syms[0]] = 1
        return torch.tensor(lens, dtype=torch.uint8)
    heap = [(f[s], i, [s]) for i, s in enumerate(syms)]
    heapq.heapify(heap)
    uid = len(heap)
    while len(heap) > 1:
        a, _, sa = heapq.heappop(heap)
        b, _, sb = heapq.heappop(heap)
        for s in sa + sb:
            lens[s] += 1
        heapq.heappush(heap, (a + b, uid, sa + sb))
        uid += 1
    if max(lens) > maxlen:
        # clamp, then lengthen the most frequent short codes until the Kraft sum fits
        for s in syms:
            lens[s] = min(lens[s], maxlen)
        kraft = sum(2.0 ** -lens[s] for s in syms)
        by_freq = sorted(syms, key=lambda s: -f[s])
        while kraft > 1.0 + 1e-12:
            for s in reversed(by_freq):  # rarest first: cheapest to lengthen
                if lens[s] < maxlen:
                    kraft -= 2.0 ** -(lens[s] + 1)
                    lens[s] += 1
                    break
    return torch.tensor(lens, dtype=torch.uint8)


def canonical_codes(lens: torch.Tensor) -> torch.Tensor:
    if lens.shape != (256,) or lens.dtype != torch.uint8 or torch.any(lens > MAXLEN):
        raise ValueError("expected 256 uint8 code lengths in 0..12")
    L = lens.tolist()
    codes = [0] * 256
    code, prev = 0, 0
    for s in sorted((s for s in range(256) if L[s] > 0), key=lambda s: (L[s], s)):
        code <<= L[s] - prev
        if code >= 1 << L[s]:
            raise ValueError("oversubscribed Huffman code lengths")
        codes[s] = code
        code += 1
        prev = L[s]
    return torch.tensor(codes, dtype=torch.int32)


def decode_table(lens: torch.Tensor, maxlen: int) -> torch.Tensor:
    _parameters(maxlen)
    codes = canonical_codes(lens).tolist()
    L = lens.tolist()
    if max(L) > maxlen:
        raise ValueError("Huffman code length exceeds maxlen")
    lut = torch.zeros(1 << maxlen, dtype=torch.int32)
    for s in range(256):
        if L[s]:
            lo = codes[s] << (maxlen - L[s])
            lut[lo : lo + (1 << (maxlen - L[s]))] = (s << 8) | L[s]
    return lut


def entropy_bits(sym: torch.Tensor) -> float:
    """Shannon entropy (bits/symbol) of a uint8 stream: the bound any order-0 coder can reach."""
    p = torch.bincount(sym.flatten().long(), minlength=256).double()
    p = p[p > 0] / p.sum()
    return float(-(p * p.log2()).sum())


def split_bytes(t: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor | None]:
    """(coded symbols, raw remainder) for a tensor, by element size."""
    b = t.contiguous().view(torch.uint8).flatten()
    es = t.element_size()
    if es == 1:
        return b, None
    if es == 2:
        b2 = b.view(-1, 2)
        return b2[:, 1].contiguous(), b2[:, 0].contiguous()  # little-endian: [1] is the high byte
    raise ValueError("Mode B codes 1- and 2-byte formats only")


def join_bytes(sym: torch.Tensor, raw: torch.Tensor | None) -> torch.Tensor:
    if raw is None:
        return sym
    return torch.stack([raw, sym], -1).flatten()


def encode(t: torch.Tensor, maxlen: int = MAXLEN, block: int = BLOCK) -> dict[str, torch.Tensor]:
    _parameters(maxlen, block)
    e = ext()
    if e is None:
        raise RuntimeError("Mode B encoding needs the C++ extension")
    sym, raw = split_bytes(t)
    lens = code_lengths(torch.bincount(sym.long(), minlength=256), maxlen)
    stream, offsets = e.huff_encode(sym, canonical_codes(lens), lens, block)
    out = {"stream": stream, "offsets": offsets, "lens": lens}
    if raw is not None:
        out["raw"] = raw
    return out


def decode(parts: dict[str, torch.Tensor], n_symbols: int, maxlen: int = MAXLEN, block: int = BLOCK,
           *, out: torch.Tensor | None = None) -> torch.Tensor:
    """Decode to flat original bytes on the stream's device.

    A contiguous uint8 ``out`` can be supplied as reusable scratch; its unused suffix is left
    unchanged. CUDA uses the optional GPU extension when available. Without that extension,
    decoding falls back to CPU and copies the result back to the input device.
    """
    _parameters(maxlen, block)
    if type(n_symbols) is not int or n_symbols < 0:
        raise ValueError("n_symbols must be nonnegative")
    for name, dtype in (("stream", torch.uint8), ("offsets", torch.int64), ("lens", torch.uint8)):
        if parts[name].dtype != dtype or parts[name].ndim != 1:
            raise ValueError(f"invalid Huffman {name} dtype or rank")
    if parts["stream"].numel() < 4 or n_symbols > (parts["stream"].numel() - 3) * 8:
        raise ValueError("Huffman output exceeds the available encoded bits")
    device = parts["stream"].device
    has_raw = "raw" in parts
    if has_raw and (parts["raw"].dtype != torch.uint8 or parts["raw"].ndim != 1 or parts["raw"].numel() != n_symbols):
        raise ValueError("raw must contain one uint8 low byte per symbol")
    nbytes = n_symbols * (2 if has_raw else 1)
    if out is not None and (out.dtype != torch.uint8 or out.device != device or
                            not out.is_contiguous() or out.numel() < nbytes):
        raise ValueError("out must be a contiguous uint8 buffer on the stream device with enough bytes")
    if device.type == "cuda":
        from ..kernels.cuda import ext as cuda_ext

        e_cuda = cuda_ext()
        if e_cuda is not None:
            result = out.view(-1)[:nbytes] if out is not None else torch.empty(nbytes, device=device, dtype=torch.uint8)
            lut = decode_table(parts["lens"].cpu(), maxlen).to(device)
            raw = parts["raw"].to(device).contiguous() if has_raw else torch.empty(0, device=device, dtype=torch.uint8)
            e_cuda.huff_decode_into(parts["stream"].contiguous(), parts["offsets"].to(device).contiguous(),
                                   n_symbols, block, lut, maxlen, raw, result)
            return result
    e = ext()
    if e is None:
        raise RuntimeError("Mode B decoding needs the C++ extension")
    cpu_parts = {k: v.cpu().contiguous() for k, v in parts.items()}
    lut = decode_table(cpu_parts["lens"], maxlen)
    sym = e.huff_decode(cpu_parts["stream"], cpu_parts["offsets"], n_symbols, block, lut, maxlen)
    result = join_bytes(sym, cpu_parts.get("raw")).to(device)
    if out is not None:
        out.view(-1)[:nbytes].copy_(result)
        return out.view(-1)[:nbytes]
    return result


def coded_bytes(parts: dict[str, torch.Tensor]) -> int:
    """Bytes the coded tensor occupies: payload, block offsets, raw low bytes, code lengths."""
    n = math.ceil(int(parts["offsets"][-1]) / 8) + parts["offsets"].numel() * 8 + parts["lens"].numel()
    if "raw" in parts:
        n += parts["raw"].numel()
    return n
