#!/usr/bin/env python3
"""Synthesize the "DO NOT RUN" demo specimen used in examples/.

Takes a real, clean GGUF and replaces ONLY its `tokenizer.chat_template`
metadata value with a hostile template, then rebuilds the file. Everything
else is preserved byte-for-byte:

  GGUF layout:  magic | version | counts | metadata KVs | tensor infos |
                <pad to alignment> | tensor data

Tensor offsets in the GGUF header are relative to the start of the tensor-data
section, so if we keep the tensor infos identical and copy the tensor-data
blocks verbatim, only the metadata section changes. We therefore re-serialize
the metadata KVs (raw value bytes preserved, so untouched KVs round-trip
exactly), keep the tensor-infos bytes verbatim, re-pad to the file's declared
alignment, and append the original tensor-data bytes unchanged.

This is a demo generator, not part of the auditor. It exists so the sample
report in examples/ is reproducible and its provenance is transparent.

Usage:
    python3 synthesize_do_not_run.py <clean.gguf> <template.j2> <out.gguf>
"""
import struct
import sys
from pathlib import Path

GGUF_MAGIC = b"GGUF"

# gguf metadata value types -> (struct fmt, size). STRING(8) and ARRAY(9) are
# variable-length and handled separately.
_SCALAR = {
    0: ("<B", 1), 1: ("<b", 1), 2: ("<H", 2), 3: ("<h", 2), 4: ("<I", 4),
    5: ("<i", 4), 6: ("<f", 4), 7: ("<B", 1), 10: ("<Q", 8), 11: ("<q", 8),
    12: ("<d", 8),
}


def _read_string(buf, off):
    (n,) = struct.unpack_from("<Q", buf, off)
    off += 8
    return buf[off:off + n], off + n  # return raw bytes, not decoded


def _skip_value(buf, off, vtype):
    """Return offset just past the value of type vtype starting at off."""
    if vtype == 8:                       # STRING
        return _read_string(buf, off)[1]
    if vtype == 9:                       # ARRAY
        (at,) = struct.unpack_from("<I", buf, off)
        (cnt,) = struct.unpack_from("<Q", buf, off + 4)
        off += 12
        for _ in range(cnt):
            off = _skip_value(buf, off, at)
        return off
    return off + _SCALAR[vtype][1]


def parse(buf):
    assert buf[:4] == GGUF_MAGIC, "not a GGUF file"
    version, = struct.unpack_from("<I", buf, 4)
    n_tensors, = struct.unpack_from("<Q", buf, 8)
    n_kv, = struct.unpack_from("<Q", buf, 16)
    off = 24

    kvs = []            # (key_bytes, vtype, raw_value_bytes)
    alignment = 32      # gguf default
    for _ in range(n_kv):
        key, off = _read_string(buf, off)
        (vtype,) = struct.unpack_from("<I", buf, off)
        off += 4
        vstart = off
        off = _skip_value(buf, off, vtype)
        raw = buf[vstart:off]
        kvs.append((key, vtype, raw))
        if key == b"general.alignment" and vtype == 4:
            (alignment,) = struct.unpack("<I", raw)
    if alignment == 0:
        alignment = 32

    infos_start = off
    for _ in range(n_tensors):
        _name, off = _read_string(buf, off)
        (nd,) = struct.unpack_from("<I", buf, off)
        off += 4 + 8 * nd + 4 + 8          # dims + type + offset
    infos = buf[infos_start:off]
    data_start = (off + alignment - 1) // alignment * alignment
    return version, n_tensors, kvs, infos, data_start, alignment


def _enc_string(s: bytes) -> bytes:
    return struct.pack("<Q", len(s)) + s


def build(path_in: Path, template: bytes, path_out: Path):
    buf = Path(path_in).read_bytes()
    version, n_tensors, kvs, infos, data_start, alignment = parse(buf)

    out = bytearray()
    out += GGUF_MAGIC
    out += struct.pack("<I", version)
    out += struct.pack("<Q", n_tensors)
    out += struct.pack("<Q", len(kvs))

    swapped = 0
    for key, vtype, raw in kvs:
        out += _enc_string(key)
        if key == b"tokenizer.chat_template":
            assert vtype == 8, "chat_template KV is not a string"
            raw = _enc_string(template)   # type 8 value = length-prefixed bytes
            swapped += 1
        out += struct.pack("<I", vtype)
        out += raw
    assert swapped == 1, f"expected exactly one tokenizer.chat_template KV, found {swapped}"

    out += infos
    while len(out) % alignment:
        out += b"\x00"
    out += buf[data_start:]               # tensor data, verbatim

    Path(path_out).write_bytes(bytes(out))
    return {"template_bytes": len(template), "tensors": n_tensors,
            "tensor_data_bytes": len(buf) - data_start,
            "size_in": len(buf), "size_out": len(out)}


if __name__ == "__main__":
    if len(sys.argv) != 4:
        sys.exit(__doc__)
    src, tpl, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    info = build(Path(src), Path(tpl).read_bytes(), Path(dst))
    print(f"wrote {dst}: {info}")
