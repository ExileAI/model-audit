#!/usr/bin/env python3
"""
model_audit — supply-chain auditor for (abliterated) model artifacts.

Audits GGUF files (metadata, chat template, tensors, sidecars) and safetensors
files (header layout, tensor sizes, index consistency, and every template
sidecar in the directory).

What it checks, per the threat model in the verification guide:
  1. File identity: SHA-256, size, GGUF version, alignment quirks.
  2. Metadata: general.* KVs, architecture sanity (name vs tensor shapes),
     quantization type vs filename claims.
  3. Chat template tampering: extracts tokenizer.chat_template from the FILE
     (not the README) and scans the Jinja for exfiltration / file access /
     sandbox-escape patterns, zero-width & homoglyph tricks.
  4. Tensor inventory: names, count, per-tensor SHA-256 (streaming), to
     enable baseline diffing against a known-good quant of the same base.
  5. Sidecars: flags any scripts/executables shipped next to the GGUF.
  6. Optional remote check: --hf <user>/<repo> compares local SHA-256 against
     the LFS hash HF publishes (reference agreement, not proof of history).

Exit codes: 0 = clean, 1 = findings, 2 = hard errors.
"""
import argparse, hashlib, json, math, os, re, struct, sys, unicodedata
from pathlib import Path
from collections import defaultdict, deque

CHUNK = 1 << 24
try:
    import gguf
except ImportError:  # safetensors auditing needs no third-party packages at all
    gguf = None

VERSION = "0.5.0"

# ---------------- findings collector ----------------
class Report:
    def __init__(self):
        self.items = []  # (severity, section, message)
        self.observations = {k: [] for k in ("structure", "templates", "provenance", "baseline")}
        self.template_coverage = []
    def observe(self, domain, status, reason, scope, required=True):
        self.observations[domain].append(dict(status=status, reason=reason, scope=scope, required=required))
    def exit_code(self):
        return 2 if any(s == "CRIT" for s, _, _ in self.items) else (1 if any(s == "WARN" for s, _, _ in self.items) else 0)
    def add(self, sev, section, msg):
        self.items.append((sev, section, msg))
    def print(self, as_json=False):
        if as_json:
            print(json.dumps([{"severity": s, "section": sec, "message": m}
                              for s, sec, m in self.items], indent=2))
            return
        order = {"CRIT": 0, "WARN": 1, "INFO": 2, "OK": 3}
        for sev, sec, msg in sorted(self.items, key=lambda x: order.get(x[0], 9)):
            print(terminal_text(f"[{sev:4}] {sec:12} {msg}"))

def terminal_text(text):
    """Keep stored evidence intact; escape controls only at the terminal boundary."""
    return "".join(c if c.isprintable() else ascii(c)[1:-1] for c in text)


def evidence_rubric(rep):
    """Summarize explicit observations, never infer coverage from finding prose."""
    domains, incomplete = {}, []
    priority = {"FAIL": 0, "NOT CHECKED": 1, "CONCERN": 2, "PASS": 3, "N/A": 4}
    for domain, observations in rep.observations.items():
        if not observations:
            domains[domain] = dict(status="NOT CHECKED", reason="Not requested or not inspected", scope=domain)
            continue
        state = min(observations, key=lambda o: priority[o["status"]])["status"]
        domains[domain] = dict(status=state,
            reason="; ".join(sorted({o["reason"] for o in observations})),
            scope="; ".join(sorted({o["scope"] for o in observations})))
        if any(o["required"] and o["status"] == "NOT CHECKED" for o in observations):
            incomplete.append(domain)
    return dict(domains=domains, coverage=dict(status="INCOMPLETE" if incomplete else "COMPLETE",
        reason=("Required checks incomplete: " + ", ".join(incomplete)) if incomplete else "Requested applicable checks completed; no safety guarantee",
        scope="Only requested, implemented checks; optional unrequested checks are excluded"))


def add_rubric(rep):
    rubric = evidence_rubric(rep)
    for domain, row in rubric["domains"].items():
        rep.add("INFO", "rubric", f"{domain}: {row['status']} — {row['reason']} (scope: {row['scope']})")
    rep.add("INFO", "rubric", "coverage: " + rubric["coverage"]["status"] + " — " + rubric["coverage"]["reason"])


def strict_json(text):
    def pairs(items):
        result = {}
        for key, value in items:
            if key in result:
                raise ValueError(f"duplicate JSON key: {key}")
            result[key] = value
        return result
    def constant(value):
        raise ValueError(f"non-finite JSON number: {value}")
    def finite_float(value):
        result = float(value)
        if not math.isfinite(result):
            raise ValueError("non-finite JSON number")
        return result
    try:
        return json.loads(text, object_pairs_hook=pairs, parse_constant=constant, parse_float=finite_float)
    except RecursionError as exc:
        raise ValueError("JSON nesting exceeds the parser limit") from exc


def json_object(path):
    obj = strict_json(Path(path).read_text("utf-8"))
    if not isinstance(obj, dict):
        raise ValueError("top level must be a JSON object")
    return obj


def valid_shape(shape):
    return isinstance(shape, list) and all(type(d) is int and d >= 0 for d in shape)


def validate_range(offset, length):
    if type(offset) is not int or type(length) is not int or offset < 0 or length < 0:
        raise ValueError("tensor range requires nonnegative integer offset and length")


# ---------------- streaming hashes ----------------
def file_hashes(path):
    sha = hashlib.sha256()
    size = 0
    with open(path, "rb") as f:
        while True:
            b = f.read(CHUNK)
            if not b:
                break
            sha.update(b)
            size += len(b)
    return sha.hexdigest(), size

def sha256_at(path, offset, length):
    """Read exactly [offset, offset+length) without loading the whole file."""
    validate_range(offset, length)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        remaining = length
        while remaining > 0:
            b = f.read(min(CHUNK, remaining))
            if not b:
                raise ValueError("short read while hashing tensor range")
            h.update(b)
            remaining -= len(b)
    return h.hexdigest()

# dtypes whose values can be widened to f32 exactly, so the same weights compare across
# a storage-dtype change. llama.cpp's converter upcasts bf16 norms to F32 during GGUF
# conversion: the bytes change, the values do not, and a byte-only comparison would call
# an honest conversion "different".
EXPANDABLE = {"BF16": 2, "F16": 2, "F32": 4}
# BF16 bit shifts and F32 identity preserve all bits, including NaN payloads.
# F16 uses struct conversion and is deliberately outside this stronger contract.
VALUE_NORMALIZATION = "f32-le-bits-v1"
BIT_PRESERVING = {"BF16", "F32"}

def _to_f32_bytes(chunk, dtype):
    """Widen a chunk of BF16/F16/F32 little-endian bytes to f32, exactly."""
    if dtype not in EXPANDABLE or len(chunk) % EXPANDABLE[dtype]:
        raise ValueError("unsupported dtype or misaligned floating-point bytes")
    if dtype == "F32":
        return chunk
    if dtype == "BF16":
        # bf16 is the top 16 bits of an f32, so widening is a byte interleave — and it can
        # be done with slice assignment at C speed instead of a Python loop per element.
        out = bytearray(len(chunk) * 2)
        out[2::4] = chunk[0::2]
        out[3::4] = chunk[1::2]
        return bytes(out)
    n = len(chunk) // 2
    return struct.pack(f"<{n}f", *struct.unpack(f"<{n}e", chunk[:2 * n]))

def tensor_fingerprints(path, offset, length, dtype):
    """Return (bytes_sha256, value_sha256_or_None) in one streaming pass.

    value_sha256 is the hash of the tensor's values widened to f32, or None for dtypes
    that cannot be widened (quantized blocks). For F32 tensors it equals the byte hash.
    """
    validate_range(offset, length)
    if dtype in EXPANDABLE and length % EXPANDABLE[dtype]:
        raise ValueError("misaligned floating-point tensor range")
    h_bytes = hashlib.sha256()
    h_vals = hashlib.sha256() if dtype in EXPANDABLE else None
    step = EXPANDABLE.get(dtype, 1)
    with open(path, "rb") as f:
        f.seek(offset)
        remaining, leftover = length, b""
        while remaining > 0:
            b = f.read(min(CHUNK, remaining))
            if not b:
                raise ValueError("short read while fingerprinting tensor range")
            remaining -= len(b)
            h_bytes.update(b)
            if h_vals is not None:
                b = leftover + b
                cut = len(b) - (len(b) % step)
                h_vals.update(_to_f32_bytes(b[:cut], dtype))
                leftover = b[cut:]
        if h_vals is not None and leftover:
            raise ValueError("incomplete floating-point element")
    return h_bytes.hexdigest(), (h_vals.hexdigest() if h_vals else None)

# ---------------- template tamper scan ----------------
DANGEROUS = [
    (r"import\s|\bimport\b",            "jinja import"),
    (r"__class__|__subclasses__|__globals__|__builtins__|__init__", "python dunder traversal (SSTI escape)"),
    (r"\battr\b|\bsetattr\b|\bgetattr\b", "dynamic attribute access"),
    (r"\bopen\b|\bread\b|\bwrite\b|\bfile\b", "file I/O wording"),
    (r"\bos\.|\bpathlib\b|\bshutil\b|\bsubprocess\b|\bexec\b|\beval\b", "exec/OS wording"),
    (r"\benvironment\b|\bconfig\b|\bself\.", "environment/config access"),
    (r"\bnamespace\s*\(\s*[^)]*\b(?:os|open|exec|eval|__)", "suspicious namespace() usage"),
    (r"\blipsum\b|\bcycler\b|\bjoiner\b", "jinja global enumeration"),
    (r"\{%(-)?\s*script", "script tag in jinja"),
    (r"tool_call\s*\(|function_call\s*\(", "template fabricates a tool call"),
    (r"\.get\s*\(\s*['\"](?:api[_-]?key|token|secret|password|authorization)", "credential-harvesting lookup"),
    (r"['\"](?:api[_-]?key|apikey|secret|password|bearer|authorization)['\"]", "credential-related string"),
]
COMMENT = re.compile(r"\{#.*?#\}", re.S)
ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff", "\u00ad"}
CONFUSABLES = set("асеорхукοαεορتشرهع٠١٢٣۴٥۶۷۸۹")  # cyr/greek/arabic/persian lookalikes

def scan_template(tpl, rep):
    if tpl is None:
        rep.add("CRIT", "template", "No tokenizer.chat_template in file — loaders fall back to"
                                      " card config or defaults; behavior not pinned by this artifact.")
        return
    if not isinstance(tpl, str):
        rep.add("WARN", "template", "template invalid: expected text")
        return
    zwc = [c for c in tpl if c in ZERO_WIDTH]
    if zwc:
        rep.add("CRIT", "template", f"{len(zwc)} zero-width/invisible chars embedded in template "
                                    f"({', '.join(hex(ord(c)) for c in sorted(set(zwc)))})")
    conf = sorted({c for c in tpl if unicodedata.normalize("NFKC", c) != c or c in CONFUSABLES
                   and not c.isascii()})
    if conf:
        rep.add("WARN", "template", f"non-ASCII confusable chars present: "
                                    f"{', '.join(f'{c} U+{ord(c):04X}' for c in conf[:8])}")
    low = tpl
    code_only = COMMENT.sub("", tpl)  # comments demoted: pattern hits there are INFO
    for pat, label in DANGEROUS:
        for m in re.finditer(pat, code_only, re.I):
            ctx = code_only[max(0, m.start()-30):m.end()+30].replace("\n", " ")
            sev = "INFO" if label.endswith("wording") and "tool" in code_only[max(0,m.start()-200):m.start()+200].lower() else "WARN"
            rep.add(sev, "template", f"{label}: ...{ctx}...")
    # structural surprise: template doing arithmetic/loops over non-message vars
    # Macro parameters, loop targets and {% set %} names are locally bound — a real
    # 300-line tool-calling template otherwise produces nothing but this INFO noise,
    # and INFO noise is how users learn to ignore a scanner.
    locals_ = set()
    for params in re.findall(r"\{%-?\s*macro\s+\w+\s*\(([^)]*)\)", tpl):
        for p in params.split(","):
            p = p.split("=")[0].strip()
            if p.isidentifier():
                locals_.add(p)
    for target in re.findall(r"\{%-?\s*for\s+(.+?)\s+in\s", tpl):
        for v in target.split(","):
            v = v.strip()
            if v.isidentifier():
                locals_.add(v)
    locals_.update(re.findall(r"\{%-?\s*set\s+(\w+)", tpl))
    locals_.update(re.findall(r"\{%-?\s*macro\s+(\w+)", tpl))  # macros the template defines itself
    for var in set(re.findall(r"\{\{\s*([a-zA-Z_][\w\.]*)", tpl)):
        if var.split(".")[0] in locals_:
            continue
        if var.split(".")[0] not in {
            "messages", "bos_token", "eos_token", "add_generation_prompt",
            "system_message", "system_prompt", "tools", "documents", "prompt",
            "strip", "tojson", "messages_json"}:
            rep.add("INFO", "template", f"references non-standard variable: {var}")


# ---------------- architecture sanity ----------------
EXPECTED_PREFIX = {"llama": "blk", "qwen2": "blk", "gemma": None}  # loose; count-checked below

def check_meta(md, path, rep):
    arch = md.get("general.architecture", "?")
    name = md.get("general.name", "?")
    quant = md.get("general.file_type")
    rep.add("INFO", "meta", f"architecture={arch} name={name!r} file_type={quant}")
    if arch == "?":
        rep.add("WARN", "meta", "no general.architecture KV — nonstandard or stripped build")
    # quant type vs filename
    # ggml General.file_type values (GGML_FTYPE_* in gguf.h). A wrong entry here does not
    # fail loudly — it silently mislabels a quant, which then feeds the quant-vs-filename
    # check. MOSTLY_BF16 is 32; 30 is IQ1_M.
    ft_names = {0: "F32", 1: "F16", 2: "Q4_0", 3: "Q4_1", 7: "Q8_0", 8: "Q5_0", 9: "Q5_1",
                10: "Q2_K", 11: "Q3_K_S", 12: "Q3_K_M", 13: "Q3_K_L", 14: "Q4_K_S", 15: "Q4_K_M",
                16: "Q5_K_S", 17: "Q5_K_M", 18: "Q6_K", 19: "IQ2_XXS", 20: "IQ2_XS", 21: "IQ3_XXS",
                23: "IQ1_S", 24: "IQ4_NL", 25: "IQ3_S", 26: "IQ3_M", 27: "IQ2_S", 28: "IQ2_M",
                29: "IQ4_XS", 30: "IQ1_M", 32: "BF16", 34: "TQ1_0", 35: "TQ2_0", 38: "MXFP4_MOE"}
    fname_q = re.findall(r"(F16|BF16|Q8_0|Q6_K|Q5_K_M|Q5_K_S|Q4_K_M|Q4_K_S|Q3_K_[SML]|Q4_0|Q4_1|Q5_0|Q5_1|Q2_K|IQ\d_S?_?[XSML]*|MXFP4)",
                         path.name.upper())
    # file_type arrives as a single-element list from the reader; unwrap ints from lists
    if isinstance(quant, list) and quant:
        quant = quant[0]
    if isinstance(quant, int) and quant in ft_names and fname_q:
        claimed = fname_q[-1]
        actual = ft_names[quant]
        if actual.upper() not in claimed and claimed not in actual:
            rep.add("WARN", "meta", f"filename says {claimed} but metadata file_type is {actual}")
    quant_label = ft_names.get(quant) if isinstance(quant, int) else None
    rep.add("INFO", "meta", f"quant={quant_label or 'unknown'}")
    for key in ("general.source.url", "general.base_model", "general.quantized_by"):
        if key in md:
            rep.add("INFO", "meta", f"{key} = {md[key]}")
    # tokenizer sanity
    if "tokenizer.ggml.model" not in md and arch != "?":
        rep.add("WARN", "meta", "no tokenizer.ggml.model — tokenizer KVs missing?")
    return arch

ATTN_TOKENS = {"gguf": ("attn_q", "attn_k", "attn_v"),
               "hf": ("q_proj", "k_proj", "v_proj", "query", "key", "value")}
ABLIT_SITES = ("ffn_down", "ffn_out", "attn_o", "down_proj", "o_proj", "out_proj", "output.weight")

def check_tensors(tensors, arch, rep, naming="gguf"):
    names = [t.name for t in tensors]
    n = len(names)
    rep.add("INFO", "tensors", f"{n} tensors total")
    if naming == "hf":
        counts, params = {}, 0
        for t in tensors:
            counts[t.dtype] = counts.get(t.dtype, 0) + 1
            numel = 1
            for d in t.shape:
                numel *= d
            params += numel
        census = ", ".join(f"{d}:{c}" for d, c in sorted(counts.items(), key=lambda x: -x[1]))
        rep.add("INFO", "tensors", f"dtypes: {census}; {params:,} parameters in this file")
    # classic structure spot-checks
    has_attn = sum(1 for x in names if any(k in x for k in ATTN_TOKENS.get(naming, ATTN_TOKENS["gguf"])))
    if has_attn and has_attn < 3:
        rep.add("WARN", "tensors", "attention projections partially missing — possible tensor pruning/swap")
    if naming == "gguf":  # safetensors empty tensors are already reported by st_tensors
        for t in tensors:
            shp = list(t.shape)
            if shp and any(s == 0 for s in shp):
                rep.add("CRIT", "tensors", f"tensor {t.name} has a zero dimension {t.shape}")
    # FFN/ATTN norm ratio sanity per layer (abliteration edits ffn_down/out or attn oproj)
    sites = [t for t in tensors if any(k in t.name for k in ABLIT_SITES)]
    if sites:
        rep.add("INFO", "tensors", f"{len(sites)} tensors sit at common abliteration edit sites "
                                   "(ffn_down/ffn_out/attn_o, or down_proj/o_proj in HF naming) — "
                                   "diff these against a trusted build")
    return names

# ---------------- sidecars ----------------
def check_sidecars(path, rep):
    for sib in sorted(path.parent.iterdir()):
        if sib == path:
            continue
        ext = sib.suffix.lower()
        if ext in {".py", ".sh", ".bash", ".ps1", ".bat", ".exe", ".dll", ".so", ".js"}:
            rep.add("WARN", "sidecar", f"executable/script shipped alongside: {sib.name}")
        elif ext in {".json", ".txt", ".md", ".yaml", ".yml"} and "config" in sib.name.lower():
            rep.add("INFO", "sidecar", f"config file present (card-vs-file mismatch possible): {sib.name}")

# ---------------- safetensors ----------------
# Layout: an 8-byte little-endian header length, that many bytes of JSON
# ({tensor_name: {dtype, shape, data_offsets}}, plus an optional "__metadata__"
# object), then the raw tensor bytes. data_offsets are relative to the start of
# the data section, so per-tensor hashing needs no tensor walk at all.
#
# Two things matter for the threat model here. One, everything in the header is
# attacker-controlled text the uploader wrote — a header agreeing with itself and
# with the file length is a consistency check, not a safety check. Two, this
# format does NOT carry the chat template: it lives in a sidecar
# (chat_template.jinja, or the chat_template key in tokenizer_config.json). That
# sidecar is the tempting target, because swapping 2 KB next to a 15 GB file that
# nobody re-hashes is far cheaper than tampering with the weights themselves.
ST_MAX_HEADER = 256 << 20
ST_DTYPE_SIZE = {"F64": 8, "I64": 8, "U64": 8, "F32": 4, "I32": 4, "U32": 4,
                 "F16": 2, "BF16": 2, "I16": 2, "U16": 2, "I8": 1, "U8": 1,
                 "BOOL": 1, "F8_E4M3": 1, "F8_E5M2": 1, "F8_E8M0": 1}

class STTensor:
    """Duck-types the gguf tensor objects the rest of the scanner works with."""
    __slots__ = ("name", "shape", "dtype", "data_offset", "n_bytes")
    def __init__(self, name, shape, dtype, data_offset, n_bytes):
        self.name, self.shape, self.dtype = name, tuple(shape), dtype
        self.data_offset, self.n_bytes = data_offset, n_bytes

def looks_like_safetensors(path):
    """Cheap magic sniff: a plausible 8-byte length followed by JSON opening with '{'.

    The length is not bounded here on purpose — an absurd length is exactly what
    read_st_header should report as a finding, not what should make us call the file
    unrecognized.
    """
    try:
        with open(path, "rb") as f:
            raw = f.read(8)
            if len(raw) < 8 or raw[:4] == b"GGUF":
                return False
            n = struct.unpack("<Q", raw)[0]
            if n < 2:
                return False
            head = f.read(64).lstrip()
    except OSError:
        return False
    return head[:1] == b"{"

def read_st_header(path, rep):
    """Parse the safetensors header. Returns (header, data_start, file_size) or None."""
    size = os.path.getsize(path)
    with open(path, "rb") as f:
        raw = f.read(8)
        if len(raw) < 8:
            rep.add("CRIT", "st-header", "header unreadable: file is shorter than the 8-byte header length")
            return None
        hlen = struct.unpack("<Q", raw)[0]
        if hlen > ST_MAX_HEADER:
            rep.add("CRIT", "st-header", f"header unreadable: declared length {hlen:,} bytes is implausible "
                                         f"(> 256 MiB) — fabricated or corrupt")
            return None
        if hlen > size - 8:
            rep.add("CRIT", "st-header", f"header unreadable: declared length {hlen:,} exceeds the file size "
                                         f"{size:,} — truncated or fabricated file")
            return None
        blob = f.read(hlen)
    try:
        if len(blob) != hlen:
            raise ValueError("short header read")
        hdr = strict_json(blob.decode("utf-8"))
    except Exception as e:
        rep.add("CRIT", "st-header", f"header invalid: not valid JSON ({e})")
        return None
    if not isinstance(hdr, dict):
        rep.add("CRIT", "st-header", "header invalid: top level is not a JSON object")
        return None
    data_start = 8 + hlen
    rep.add("INFO", "st-header", f"header {hlen:,} bytes, data section starts at {data_start:,}, file {size:,}")
    if data_start % 8:
        rep.add("INFO", "st-header", f"data section starts at {data_start} — not 8-byte aligned "
                                     f"(the reference writers pad the header so it is)")
    return hdr, data_start, size

def st_tensors(hdr, data_start, size, rep):
    """Validate every declared tensor and return shims the shared checks can use."""
    rep.structure_valid = True
    rep.structure_complete = True
    rep.structure_concern = False
    rep.baseline_eligible = True
    meta = hdr.get("__metadata__")
    if "__metadata__" in hdr and (not isinstance(meta, dict) or not all(isinstance(v, str) for v in meta.values())):
        rep.structure_valid = False
        rep.baseline_eligible = False
        rep.add("CRIT", "st-header", "__metadata__ must be a string-to-string JSON object")
    elif isinstance(meta, dict):
        for k, v in sorted(meta.items()):
            if isinstance(v, str) and len(v) <= 200:
                rep.add("INFO", "provenance", f"__metadata__[{k}] = {v}")
            else:
                rep.add("INFO", "provenance", f"__metadata__[{k}] present (non-string or oversized)")
    entries = {k: v for k, v in hdr.items() if k != "__metadata__"}
    if not entries:
        rep.add("INFO", "st-header", "no tensors declared in header (empty artifact)")
    tensors, spans = [], []
    for name, ent in entries.items():
        if not isinstance(ent, dict) or not {"dtype", "shape", "data_offsets"} <= set(ent):
            rep.structure_valid = False
            rep.baseline_eligible = False
            rep.add("CRIT", "st-header", f"tensor entry missing dtype/shape/data_offsets: {name}")
            continue
        dtype, shape, off = ent["dtype"], ent["shape"], ent["data_offsets"]
        if not isinstance(dtype, str) or not valid_shape(shape):
            rep.structure_valid = False
            rep.baseline_eligible = False
            rep.add("CRIT", "st-header", f"tensor entry invalid dtype or shape: {name}")
            continue
        if not (isinstance(off, list) and len(off) == 2 and all(type(x) is int for x in off)):
            rep.structure_valid = False
            rep.baseline_eligible = False
            rep.add("CRIT", "st-header", f"tensor entry has malformed data_offsets: {name} ({off!r})")
            continue
        begin, end = off
        if begin < 0 or end < begin:
            rep.structure_valid = False
            rep.baseline_eligible = False
            rep.add("CRIT", "st-header", f"tensor entry has nonsensical offsets: {name} [{begin}, {end})")
            continue
        n_bytes = end - begin
        if data_start + end > size:
            rep.structure_valid = False
            rep.baseline_eligible = False
            rep.add("CRIT", "st-tensors", f"tensor size: {name} declares data ending at "
                                          f"{data_start + end:,}, past the end of the file ({size:,}) — truncated")
            continue
        unit = ST_DTYPE_SIZE.get(dtype)
        numel = 1
        for d in shape:
            numel *= d
        if unit is None:
            rep.structure_complete = False
            rep.baseline_eligible = False
            rep.add("WARN", "st-tensors", f"tensor size: {name} dtype {dtype!r} is not a known type — "
                                          f"its byte size cannot be verified")
        elif int(numel * unit) != n_bytes:
            rep.structure_valid = False
            rep.baseline_eligible = False
            rep.add("CRIT", "st-tensors", f"tensor size: {name} is {dtype} {list(shape)}, which implies "
                                          f"{int(numel * unit):,} bytes, but the header declares {n_bytes:,} — "
                                          f"the header contradicts itself")
        if n_bytes == 0 or any(d == 0 for d in shape):
            rep.add("WARN", "st-tensors", f"empty tensor: {name} is empty ({dtype} {list(shape)}) — "
                                         "legal in safetensors; inventory heuristic only, review whether intended")
        tensors.append(STTensor(name, shape, dtype, data_start + begin, n_bytes))
        spans.append((begin, end, name))
    spans.sort()
    prev_end = 0
    for begin, end, name in spans:
        if begin < prev_end:
            rep.structure_valid = False
            rep.baseline_eligible = False
            rep.add("CRIT", "st-tensors", f"tensor layout: {name} starts inside another tensor's data — "
                                          f"the declared layout is not valid safetensors")
        elif begin > prev_end:
            rep.structure_concern = True
            rep.baseline_eligible = False
            rep.add("WARN", "st-tensors", f"undeclared gap: {begin - prev_end:,} bytes between tensors before "
                                          f"{name} that no header entry accounts for")
        prev_end = max(prev_end, end)
    if prev_end < size - data_start:
        rep.structure_concern = True
        rep.baseline_eligible = False
        rep.add("WARN", "st-tensors", f"trailing data: {size - data_start - prev_end:,} bytes after the last "
                                      f"declared tensor that the header does not account for")
    return tensors


def check_st_config(path, rep):
    """Inspect config and tokenizer config independently, including remote-code hints."""
    for name in ("config.json", "tokenizer_config.json"):
        fp = path.parent / name
        if not fp.exists():
            if name == "config.json":
                rep.add("INFO", "meta", "no config.json in this directory — provenance is not pinned here")
            continue
        try:
            cfg = json_object(fp)
            archs = cfg.get("architectures", [])
            if not isinstance(archs, list) or not all(isinstance(x, str) for x in archs):
                raise ValueError("architectures must be a list of strings")
            if "auto_map" in cfg and (not isinstance(cfg["auto_map"], dict) or not all(
                    isinstance(v, str) or (isinstance(v, list) and all(x is None or isinstance(x, str) for x in v)) for v in cfg["auto_map"].values())):
                raise ValueError("auto_map must map names to text or lists of text/null")
            if "quantization_config" in cfg and not isinstance(cfg["quantization_config"], dict):
                raise ValueError("quantization_config must be an object")
            for key in ("model_type", "torch_dtype", "transformers_version", "_name_or_path", "name_or_path"):
                if key in cfg and cfg[key] is not None and not isinstance(cfg[key], str):
                    raise ValueError(f"{key} must be text")
        except (OSError, ValueError, UnicodeError) as e:
            rep.add("WARN", "sidecar", f"{name} present but unreadable: {e}")
            rep.observe("provenance", "NOT CHECKED", "Present configuration could not be validated", name + " self-reported configuration")
            continue
        if name == "config.json":
            model_name = cfg.get("_name_or_path") or cfg.get("name_or_path")
            rep.add("INFO", "meta", f"architecture={archs[0] if archs else cfg.get('model_type', '?')} "
                                    f"name={model_name!r} file_type={cfg.get('torch_dtype', '?')} (safetensors)")
            for key in ("model_type", "torch_dtype", "transformers_version", "base_model", "quantized_by", "converted_by", "source"):
                if cfg.get(key):
                    rep.add("INFO", "meta", f"{key} = {cfg[key]}")
            if cfg.get("quantization_config"):
                rep.add("INFO", "provenance", "config.json carries quantization_config — provenance is self-reported")
        if cfg.get("auto_map"):
            rep.add("WARN", "sidecar", f"auto_map declared in {name}: trust_remote_code may execute uploader Python")


def _scan_template_labeled(text, rep, label):
    before = len(rep.items)
    scan_template(text, rep)
    for i in range(before, len(rep.items)):
        sev, sec, msg = rep.items[i]
        rep.items[i] = (sev, sec, f"{label}: {msg}")


def record_template(text, rep, label, logical_name):
    before = len(rep.items)
    _scan_template_labeled(text, rep, label)
    suspicious = any(s in ("WARN", "CRIT") for s, _, _ in rep.items[before:])
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
    rep.template_coverage.append(dict(label=label, logical_name=logical_name, sha256=digest, state="inspected"))
    rep.observe("templates", "CONCERN" if suspicious else "PASS",
                "Signature findings require review" if suspicious else "All inspected text passed implemented signatures",
                label + " (signature scan only)")
    rep.add("INFO", "template", f"template sidecar: {label} ({len(text.encode('utf-8')):,} bytes)")


def template_incomplete(rep, label, reason):
    rep.template_coverage.append(dict(label=label, logical_name=None, sha256=None, state="incomplete"))
    rep.observe("templates", "NOT CHECKED", reason, label)
    rep.add("WARN", "template", f"template invalid: {label}: {reason}")


def scan_st_templates(path, rep):
    """Scan every supported source, preserving incomplete coverage and logical names."""
    d = path.parent
    jinjas = sorted(set(d.glob("*.jinja")) | set(d.glob("*.jinja2")) | set((d / "additional_chat_templates").glob("*.jinja")))
    for fp in jinjas:
        label = str(fp.relative_to(d))
        logical = fp.stem if fp.parent.name == "additional_chat_templates" else ("default" if fp.stem == "chat_template" else fp.stem)
        try:
            record_template(fp.read_text("utf-8"), rep, label, logical)
        except (OSError, UnicodeError) as e:
            template_incomplete(rep, label, str(e))
    for filename in ("tokenizer_config.json", "chat_template.json"):
        fp = d / filename
        if not fp.exists():
            continue
        try:
            obj = json_object(fp)
        except (OSError, ValueError, UnicodeError) as e:
            template_incomplete(rep, filename, str(e))
            continue
        if "chat_template" not in obj:
            continue
        tpl = obj["chat_template"]
        if filename == "chat_template.json" and not isinstance(tpl, str):
            template_incomplete(rep, filename, "legacy chat_template.json requires a chat_template string")
            continue
        entries = []
        if isinstance(tpl, str):
            entries = [("default", tpl, filename)]
        elif isinstance(tpl, dict):
            entries = [(key, val, f"{filename}[chat_template.{key}]") for key, val in tpl.items()]
        elif isinstance(tpl, list):
            for i, entry in enumerate(tpl):
                label = f"{filename}[chat_template[{i}]]"
                if isinstance(entry, str):
                    entries.append(("default" if i == 0 else f"legacy_{i}", entry, label))
                elif isinstance(entry, dict) and isinstance(entry.get("name"), str) and isinstance(entry.get("template"), str):
                    entries.append((entry["name"], entry["template"], label))
                else:
                    template_incomplete(rep, label, "expected text or a named {name, template} record")
        else:
            template_incomplete(rep, filename, "chat_template must be text, a map, or a list")
        if isinstance(tpl, (dict, list)) and not tpl:
            template_incomplete(rep, filename, "empty template collection")
        for logical, text, label in entries:
            if not isinstance(text, str) or not logical:
                template_incomplete(rep, label, "template name and text must be strings")
            else:
                record_template(text, rep, label, logical)
    by_name = {}
    for source in rep.template_coverage:
        if source["state"] == "inspected":
            by_name.setdefault(source["logical_name"], set()).add(source["sha256"])
    for logical, hashes in sorted(by_name.items()):
        if len(hashes) > 1:
            rep.add("WARN", "template", f"template sources for {logical!r}: they differ — loader selection is runtime-dependent")
            rep.observe("templates", "CONCERN", "Sources for the same logical template differ", logical)
    if not rep.template_coverage:
        rep.add("INFO", "template", "no chat template sidecar in this directory — the loader supplies it")
        rep.observe("templates", "NOT CHECKED", "No template source was available", "supported safetensors sidecars")

def check_st_index(path, tensors, rep):
    """Multi-shard models are described by model.safetensors.index.json — check it agrees."""
    idxs = sorted(path.parent.glob("*.safetensors.index.json"))
    if not idxs:
        return
    fp = idxs[0]
    try:
        idx = json_object(fp)
        wmap = idx.get("weight_map")
        meta = idx.get("metadata", {})
        if not isinstance(wmap, dict) or not all(isinstance(v, str) for v in wmap.values()):
            raise ValueError("weight_map must map tensor names to shard strings")
        if not isinstance(meta, dict) or ("total_size" in meta and (type(meta["total_size"]) is not int or meta["total_size"] < 0)):
            raise ValueError("metadata.total_size must be a nonnegative integer")
    except Exception as e:
        rep.add("WARN", "sidecar", f"index mismatch: {fp.name} unreadable ({e})")
        rep.observe("structure", "NOT CHECKED", "Present shard index could not be validated", fp.name)
        return
    wmap = idx.get("weight_map") or {}
    shards = sorted(set(wmap.values()))
    rep.add("INFO", "index", f"{fp.name}: {len(wmap)} tensors across {len(shards)} shard(s)")
    mine = {t.name for t in tensors}
    listed_here = {k for k, v in wmap.items() if Path(v).name == path.name}
    missing = sorted(listed_here - mine)
    unlisted = sorted(mine - listed_here)
    if missing:
        rep.observe("structure", "FAIL", "Index references tensors absent from this shard", fp.name)
        rep.add("CRIT", "index", f"index mismatch: {len(missing)} tensor(s) the index routes to this shard are "
                                 f"absent from it (e.g. {', '.join(missing[:3])})")
    if unlisted:
        rep.observe("structure", "CONCERN", "Shard tensors are not listed in selected index", fp.name)
        rep.add("WARN", "index", f"index mismatch: {len(unlisted)} tensor(s) present here are not listed in "
                                 f"{fp.name} (e.g. {', '.join(unlisted[:3])}) — added or renamed after indexing")
    total = (idx.get("metadata") or {}).get("total_size")
    if len(shards) == 1 and isinstance(total, int):
        declared = sum(t.n_bytes for t in tensors)
        if total != declared:
            rep.observe("structure", "FAIL", "Selected single-shard index total_size disagrees", fp.name)
            rep.add("CRIT", "index", f"index mismatch: metadata.total_size is {total:,} bytes but this file "
                                     f"declares {declared:,} — the index describes a different file")

# ---------------- uploader-supplied checksums ----------------
MANIFEST_NAMES = ("MANIFEST.txt", "MANIFEST", "SHA256SUMS", "SHA256SUMS.txt",
                  "checksums.txt", "checksums.sha256")

def check_manifest(path, sha, rep):
    """Some repos ship their own hashes. Verifying against them is a consistency check.

    The manifest is the uploader's own claim, so agreeing with it proves nothing about
    safety. But disagreeing with it — or having been renamed under it — is exactly the
    drift a swap or a re-upload leaves behind, and a repo that publishes hashes is
    inviting you to check.
    """
    fp = next((path.parent / n for n in MANIFEST_NAMES if (path.parent / n).is_file()), None)
    if fp is None:
        fp = next((p for p in sorted(path.parent.glob("*.sha256")) + sorted(path.parent.glob("*checksum*"))
                   if p.is_file()), None)
    if fp is None:
        return
    scope = f"Uploader-supplied checksum {fp.name}; consistency only, not independent trust"
    entries = {}
    try:
        lines = fp.read_text("utf-8").splitlines()
    except (OSError, UnicodeError) as e:
        rep.add("WARN", "manifest", f"manifest unreadable: {e}")
        rep.observe("provenance", "NOT CHECKED", "Checksum list could not be read", scope)
        return
    unsupported = False
    for line in lines:
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.split()
        if len(parts) != 2:
            unsupported = True
            continue
        a, b = parts[0], parts[-1].lstrip("*")
        if re.fullmatch(r"[0-9a-fA-F]{64}", a):
            entries.setdefault(b, []).append(a.lower())
        elif re.fullmatch(r"[0-9a-fA-F]{64}", b):
            entries.setdefault(a, []).append(b.lower())
        else:
            unsupported = True
    if unsupported:
        rep.add("WARN", "manifest", f"ambiguous manifest: {fp.name} contains unsupported records")
        rep.observe("provenance", "CONCERN", "Unsupported records prevent an unambiguous selected checksum", scope)
        return
    if not entries:
        rep.add("WARN", "manifest", f"{fp.name} present but no readable SHA-256 lines")
        rep.observe("provenance", "NOT CHECKED", "No usable checksums", scope)
        return
    if path.name in entries:
        hashes = entries[path.name]
        if len(hashes) != 1:
            rep.add("WARN", "manifest", f"ambiguous manifest: duplicate entries for {path.name}")
            rep.observe("provenance", "CONCERN", "Duplicate selected filename entries", scope)
        elif hashes[0] == sha:
            rep.add("OK", "manifest", f"SHA-256 matches the uploader's own entry for {path.name} in {fp.name}")
            rep.observe("provenance", "PASS", "Selected unique filename checksum matches", scope)
        else:
            rep.add("CRIT", "manifest", f"MANIFEST MISMATCH: {path.name} differs from its selected checksum")
            rep.observe("provenance", "FAIL", "Selected unique filename checksum differs", scope)
        return
    names = [name for name, hashes in entries.items() if sha in hashes]
    if names:
        rep.add("INFO", "manifest", f"renamed since the manifest was written: matching content is listed as {', '.join(sorted(names))}; naming history is not independently verified")
        rep.observe("provenance", "CONCERN", "Content matches another name, not the selected filename", scope)
        return
    rep.add("INFO", "manifest", f"{fp.name} present ({len(entries)} entries) but this file is not listed in it")
    rep.observe("provenance", "NOT CHECKED", "Checksum list does not cover this filename", scope, required=False)

# ---------------- HF remote check ----------------
def hf_check(repo, local_sha, path, rep, revision=None):
    scope = f"HF {repo}; repository consistency only, not uploader trust"
    try:
        from huggingface_hub import HfApi
        info = HfApi().model_info(repo, files_metadata=True, revision=revision)
        scope += " at resolved commit " + str(info.sha)
        rep.add("INFO", "remote", f"resolved revision: {info.sha} (requested {revision or 'default branch'})")
        matches = [s for s in info.siblings if s.lfs and s.lfs.sha256 == local_sha]
        byname = [s for s in info.siblings if s.lfs and Path(s.rfilename).name == path.name]
        if len(matches) == 1:
            rep.add("OK", "remote", f"local SHA-256 matches HF LFS for {repo}@{info.sha} ({matches[0].rfilename})")
            rep.observe("provenance", "PASS", "One repository LFS entry matches the selected file hash", scope)
        elif len(matches) > 1 or len(byname) > 1:
            rep.add("WARN", "remote", "ambiguous remote reference: multiple matching hashes or basenames")
            rep.observe("provenance", "CONCERN", "Remote reference is ambiguous", scope)
        elif len(byname) == 1:
            rep.add("CRIT", "remote", f"SHA-256 MISMATCH vs {repo}@{info.sha}: {byname[0].rfilename}")
            rep.observe("provenance", "FAIL", "Unique basename reference has a different hash", scope)
        else:
            rep.add("WARN", "remote", f"no file with this SHA-256 or filename in {repo}")
            rep.observe("provenance", "NOT CHECKED", "No applicable LFS reference found", scope)
    except Exception as e:
        rep.add("WARN", "remote", f"HF fetch failed: {e}")
        rep.observe("provenance", "NOT CHECKED", "Remote lookup failed", scope)

# ---------------- baseline tensor hash dump / compare ----------------
def dump_tensor_hashes(tensors, path, byte_order="little"):
    """Per-tensor byte hashes, plus value hashes where widening to f32 is exact."""
    out, vals = {}, {}
    for t in tensors:
        dtype = getattr(t, "dtype", None) or t.tensor_type.name
        # Raw on-disk GGUF bytes may be big-endian. Do not interpret those as LE.
        hb, hv = tensor_fingerprints(path, t.data_offset, t.n_bytes,
                                     dtype if byte_order == "little" else None)
        out[t.name] = hb
        if hv:
            vals[t.name] = hv
    return out, vals

def write_baseline(tensors, path, sha, fmt, rep, baseline_out=None, byte_order=None):
    # safetensors is little-endian by definition; GGUF must supply reader evidence.
    byte_order = "little" if fmt == "safetensors" else byte_order
    hashes, values = dump_tensor_hashes(tensors, path, byte_order)
    shapes = {t.name: [int(d) for d in t.shape] for t in tensors}
    value_metadata = {t.name: dict(dtype=getattr(t, "dtype", None) or t.tensor_type.name,
                                  byte_order=byte_order, normalization=VALUE_NORMALIZATION)
                      for t in tensors if t.name in values
                      and (getattr(t, "dtype", None) or t.tensor_type.name) in BIT_PRESERVING}
    outp = Path(baseline_out) if baseline_out is not None else path.with_suffix(path.suffix + ".tensorhashes.json")
    # Only an exclusive-open conflict is a recoverable output warning. Other I/O
    # errors still propagate, and previously collected critical findings survive.
    try:
        stream = outp.open("x", encoding="utf-8")
    except FileExistsError:
        rep.add("WARN", "baseline", f"baseline exists: {outp} — pass --baseline-out to write a new snapshot; existing file unchanged")
        return
    try:
        with stream:
            json.dump({"file_sha256": sha, "format": fmt, "shapes": shapes,
                       "tensors": hashes, "values": values,
                       "value_metadata": value_metadata}, stream, indent=1)
    except Exception:
        outp.unlink()
        raise
    rep.add("OK", "baseline", f"per-tensor SHA-256 written to {outp}")

# ---------------- main ----------------
def _audit_gguf(path, rep, sha, size, do_hashes, hf_repo, hf_revision, baseline_out=None):
    rep.add("INFO", "file", "format=gguf")
    if gguf is None:
        rep.observe("structure", "NOT CHECKED", "GGUF dependency unavailable", "GGUF artifact")
        rep.observe("templates", "NOT CHECKED", "GGUF could not be inspected", "GGUF embedded template only")
        rep.add("CRIT", "file", "the gguf python package is not installed (pip install gguf) — "
                                "GGUF files cannot be parsed without it")
        return
    try:
        rdr = gguf.GGUFReader(str(path))
    except Exception as e:
        rep.observe("structure", "NOT CHECKED", "GGUF parser failed", "GGUF artifact")
        rep.observe("templates", "NOT CHECKED", "GGUF could not be inspected", "GGUF embedded template only")
        rep.add("CRIT", "file", f"GGUF header unreadable: {e}")
        return
    md = {}
    for k in rdr.fields:
        field = rdr.fields[k]
        try:
            ftype = gguf.GGUFValueType(field.types[0])
            if ftype == gguf.GGUFValueType.STRING:
                md[k] = str(bytes(field.parts[field.data[0]]), "utf-8", "replace")
            elif ftype == gguf.GGUFValueType.ARRAY:
                inner = gguf.GGUFValueType(field.types[1])
                if inner == gguf.GGUFValueType.STRING:
                    md[k] = [str(bytes(field.parts[i]), "utf-8", "replace") for i in field.data]
                else:
                    md[k] = f"<array of {inner.name}, len {len(field.data)}>"
            else:
                md[k] = field.parts[field.data[-1]].tolist()
        except Exception as e:
            md[k] = f"<unreadable: {e}>"
    start = len(rep.items)
    arch = check_meta(md, path, rep)
    tpl = md.get("tokenizer.chat_template")
    if tpl is None and arch == "clip":
        rep.add("INFO", "template", "No tokenizer.chat_template — normal for vision projector (mmproj) files")
        rep.observe("templates", "N/A", "Vision projector has no chat template by design", "GGUF embedded template only")
    elif isinstance(tpl, str):
        record_template(tpl, rep, "GGUF embedded tokenizer.chat_template", "default")
    else:
        scan_template(tpl, rep)
        rep.observe("templates", "NOT CHECKED", "Embedded chat template missing or invalid", "GGUF embedded template only")
    tensors = list(rdr.tensors)
    check_tensors(tensors, arch, rep)
    invalid = any(sev == "CRIT" and sec != "template" for sev, sec, _ in rep.items[start:])
    rep.observe("structure", "FAIL" if invalid else "PASS", "GGUF reader and implemented tensor checks only", "GGUF structural checks, not all format invariants")
    check_sidecars(path, rep)
    check_manifest(path, sha, rep)
    if do_hashes and not invalid:
        byte_order = "little" if rdr.endianess == gguf.GGUFEndian.LITTLE else "big"
        write_baseline(tensors, path, sha, "gguf", rep, baseline_out, byte_order)
    if hf_repo:
        hf_check(hf_repo, sha, path, rep, revision=hf_revision)

def _audit_st(path, rep, sha, size, do_hashes, hf_repo, hf_revision, baseline_out=None):
    rep.add("INFO", "file", "format=safetensors")
    parsed = read_st_header(path, rep)
    if parsed is None:
        rep.observe("structure", "FAIL", "Invalid or truncated header", "safetensors header")
        rep.observe("templates", "NOT CHECKED", "Artifact parsing failed before template inspection", "supported safetensors sidecars")
        return
    hdr, data_start, fsize = parsed
    start = len(rep.items)
    tensors = st_tensors(hdr, data_start, fsize, rep)
    invalid = not rep.structure_valid
    incomplete = not rep.structure_complete
    warned = rep.structure_concern
    state = "FAIL" if invalid else ("NOT CHECKED" if incomplete else ("CONCERN" if warned else "PASS"))
    rep.observe("structure", state, "Implemented header and tensor-layout checks" + (" incomplete for unsupported dtype" if incomplete else " completed"), "safetensors file only")
    if invalid and incomplete:
        rep.observe("structure", "NOT CHECKED", "Unsupported dtype prevents complete size validation", "safetensors tensor sizes")
    check_tensors(tensors, "?", rep, naming="hf")
    check_st_config(path, rep)
    scan_st_templates(path, rep)
    check_st_index(path, tensors, rep)
    check_sidecars(path, rep)
    check_manifest(path, sha, rep)
    structure_complete = rep.baseline_eligible and all(
        o["status"] in ("PASS", "N/A") for o in rep.observations["structure"])
    if do_hashes and structure_complete:
        write_baseline(tensors, path, sha, "safetensors", rep, baseline_out)
    elif do_hashes:
        rep.add("CRIT", "baseline", "baseline refused: tensor structure was not fully validated")
    if hf_repo:
        hf_check(hf_repo, sha, path, rep, revision=hf_revision)

def audit(path, rep, do_hashes=False, hf_repo=None, hf_revision=None, baseline_out=None):
    rep.add("INFO", "file", f"{path}")
    sha, size = file_hashes(path)
    rep.add("INFO", "file", f"sha256={sha}")
    rep.add("INFO", "file", f"size={size:,}")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"GGUF":
        return _audit_gguf(path, rep, sha, size, do_hashes, hf_repo, hf_revision, baseline_out)
    if looks_like_safetensors(path):
        return _audit_st(path, rep, sha, size, do_hashes, hf_repo, hf_revision, baseline_out)
    rep.observe("structure", "FAIL", "Unrecognized artifact format", "file format")
    rep.observe("templates", "NOT CHECKED", "Artifact format not recognized", "template scan")
    rep.add("CRIT", "file", f"bad magic {magic!r} — not a GGUF or safetensors file")

def normalized_fingerprint(base, name):
    """Return only a self-described F32 digest; never interpret an untyped byte hash."""
    meta = base.get("value_metadata", {}).get(name, {})
    if (meta.get("dtype") in BIT_PRESERVING and meta.get("byte_order") == "little"
            and meta.get("normalization") == VALUE_NORMALIZATION):
        return base.get("values", {}).get(name)
    return None


def match_blobs(A, B):
    """Consume once: byte matches, typed F32 matches, then ambiguous candidates.

    The four-value return stays compatible; callers can identify typed pairs with
    normalized_fingerprint(). Metadata describes a hash, not its producer's trust.
    """
    free_a, free_b = set(A["tensors"]), set(B["tensors"])
    exact, candidates = [], []
    buckets = defaultdict(deque)
    for name in sorted(free_a):
        buckets[A["tensors"][name]].append(name)
    for name in sorted(free_b):
        bucket = buckets[B["tensors"][name]]
        if bucket:
            source = bucket.popleft()
            exact.append((source, name)); free_a.remove(source)
    free_b.difference_update(name for _, name in exact)
    # Match typed value-to-value evidence before ambiguous legacy hits can take
    # its source. A raw integer digest is never promoted to an F32 fingerprint.
    buckets = defaultdict(deque)
    for name in sorted(free_a):
        digest = normalized_fingerprint(A, name)
        if digest is not None:
            buckets[digest].append(name)
    for name in sorted(free_b):
        digest = normalized_fingerprint(B, name)
        if digest is not None and buckets[digest]:
            source = buckets[digest].popleft()
            candidates.append((source, name)); free_a.remove(source)
    free_b.difference_update(name for _, name in candidates)
    buckets = defaultdict(deque)
    for name in sorted(free_a):
        for digest in sorted({A["tensors"][name], A.get("values", {}).get(name)} - {None}):
            buckets[digest].append(name)
    for name in sorted(free_b):
        for digest in sorted({B["tensors"][name], B.get("values", {}).get(name)} - {None}):
            bucket = buckets[digest]
            while bucket and bucket[0] not in free_a:
                bucket.popleft()
            if bucket:
                source = bucket.popleft()
                candidates.append((source, name)); free_a.remove(source)
                break
    free_b.difference_update(name for _, name in candidates)
    return exact, candidates, sorted(free_a), sorted(free_b)


def blob_identity(A, B):
    """Compatibility tuple; legacy hashes cannot establish numeric or shape agreement."""
    exact, candidates, _, unmatched = match_blobs(A, B)
    return len(exact), len(candidates), len(B["tensors"]), 0, 0, unmatched


def read_baseline(path):
    base = json_object(path)
    digest = lambda h: isinstance(h, str) and re.fullmatch(r"[0-9a-fA-F]{64}", h) is not None
    if not isinstance(base.get("tensors"), dict) or not all(digest(v) for v in base["tensors"].values()):
        raise ValueError("baseline tensors must map names to SHA-256 digests")
    if "file_sha256" in base and not digest(base["file_sha256"]):
        raise ValueError("invalid file_sha256")
    if "format" in base and not isinstance(base["format"], str):
        raise ValueError("baseline format must be text")
    for key, validator in (("values", digest), ("shapes", valid_shape)):
        if key not in base:
            continue
        if not isinstance(base[key], dict) or not set(base[key]) <= set(base["tensors"]) or not all(validator(v) for v in base[key].values()):
            raise ValueError(f"invalid baseline {key} mapping")
    if "value_metadata" in base:
        metadata = base["value_metadata"]
        if (not isinstance(metadata, dict) or not set(metadata) <= set(base.get("values", {}))
                or not all(isinstance(v, dict) and all(isinstance(v.get(k), str)
                    for k in ("dtype", "byte_order", "normalization")) for v in metadata.values())):
            raise ValueError("invalid baseline value_metadata mapping")
    for key in ("tensors", "values"):
        if key in base:
            base[key] = {name: value.lower() for name, value in base[key].items()}
    for name, meta in base.get("value_metadata", {}).items():
        if (meta["dtype"] == "F32" and normalized_fingerprint(base, name) is not None
                and base["values"][name] != base["tensors"][name]):
            raise ValueError("F32 little-endian value hash must equal its byte hash")
    return base


def diff_baselines(file_a, file_b, rep):
    A, B = read_baseline(file_a), read_baseline(file_b)
    ta, tb = A["tensors"], B["tensors"]
    rep.add("INFO", "diff", f"A: {file_a}; B: {file_b}")
    scope = "Recorded tensor bytes and shapes only; dtype, byte order, semantic roles and trust NOT VERIFIED"
    fa, fb = A.get("format"), B.get("format")
    if fa not in ("gguf", "safetensors") or fb not in ("gguf", "safetensors") or fa != fb:
        scope = "Recorded byte and typed normalized-F32 fingerprints only; cross-format shapes, semantic roles, model equivalence, history and trust NOT VERIFIED"
        rep.add("INFO", "diff", "comparing across formats or legacy unknown formats: matching content does not establish tensor-role mapping or conversion history")
        exact, candidates, only_a, only_b = match_blobs(A, B)
        normalized = [(a, b) for a, b in candidates if normalized_fingerprint(A, a) is not None
                      and normalized_fingerprint(A, a) == normalized_fingerprint(B, b)]
        ambiguous = len(candidates) - len(normalized)
        rep.add("INFO", "diff", f"content match: {len(exact)} of {len(tb)} B blobs are byte-identical; {len(normalized)} exact normalized-F32 fingerprint matches; {ambiguous} normalized-hash candidates (INCONCLUSIVE); {len(only_a)} unmatched in A, {len(only_b)} unmatched in B")
        if ambiguous:
            rep.add("WARN", "diff", "normalized-hash candidates lack compatible typed normalization metadata: numeric equivalence and lossless conversion are not established")
        if not ta or not tb:
            rep.add("WARN", "diff", "baseline comparison incomplete: empty tensor inventory")
        rep.add("INFO", "diff", "Nonmatches are not comparable this way; quantization can change values by design. No model identity, numeric equivalence, role mapping or tampering conclusion follows.")
        for side, names in (("A", only_a), ("B", only_b)):
            for name in names[:8]:
                rep.add("INFO", "diff", f"not matched in {side}: {name}")
        full = bool(ta and tb) and not (only_a or only_b or ambiguous) and {fa, fb} <= {"gguf", "safetensors"}
        rep.observe("baseline", "PASS" if full else "CONCERN",
                    "Both nonempty inventories agree on recorded byte or typed normalized-F32 fingerprints" if full else
                    "Cross-format or legacy comparison is inconclusive, including ambiguous, empty or subset matches", scope)
        return
    common = set(ta) & set(tb)
    only_a, only_b = sorted(set(ta) - set(tb)), sorted(set(tb) - set(ta))
    changed = sorted(k for k in common if ta[k] != tb[k])
    sa, sb = A.get("shapes", {}), B.get("shapes", {})
    missing = sorted(k for k in common if k not in sa or k not in sb)
    reshaped = sorted(k for k in common if k in sa and k in sb and sa[k] != sb[k])
    for name in only_a:
        rep.add("WARN", "diff", f"tensor only in A (removed in B): {name}")
    for name in only_b:
        rep.add("WARN", "diff", f"tensor only in B (added/renamed): {name}")
    for name in changed:
        rep.add("CRIT", "diff", f"TENSOR CONTENT CHANGED: {name} — cause not determined")
    for name in reshaped:
        rep.add("CRIT", "diff", f"TENSOR SHAPE CHANGED: {name} — recorded shape differs")
    if missing or not ta or not tb:
        rep.add("WARN", "diff", "baseline comparison incomplete: missing shape metadata or empty tensor set")
    differences = bool(changed or reshaped or only_a or only_b)
    status = "FAIL" if differences else ("CONCERN" if missing or not ta else "PASS")
    rep.observe("baseline", status, "Recorded bytes/shapes differ" if differences else ("Legacy or empty baseline is inconclusive" if status == "CONCERN" else "Both tensor sets agree on recorded bytes and shapes"), scope)
    rep.add("INFO", "diff", scope)
    if status == "PASS":
        rep.add("OK", "diff", "baselines agree on recorded tensor bytes and shapes; this does not prove safety or model equivalence")


def cmd_diff(args):
    return run_cli_report(lambda rep: diff_baselines(args.baseline_a, args.baseline_b, rep), args.json, "baseline")


def run_cli_report(operation, as_json, domain):
    rep = Report()
    try:
        operation(rep)
    except (OSError, ValueError, TypeError, OverflowError) as e:
        rep.add("CRIT", "audit", f"input error: {e}")
        rep.observe(domain, "NOT CHECKED", "Requested operation could not complete", "input/output error")
    add_rubric(rep)
    rep.print(as_json=as_json)
    return rep.exit_code()


def main():
    ap = argparse.ArgumentParser(description="audit a GGUF or safetensors file for supply-chain tampering")
    ap.add_argument("--version", action="version", version=f"model-audit {VERSION}")
    ap.add_argument("model", type=Path, nargs="?", help="model file to audit (GGUF or safetensors; format auto-detected)")
    ap.add_argument("--tensor-hashes", action="store_true", help="emit a new per-tensor SHA-256 baseline JSON; never overwrite")
    ap.add_argument("--baseline-out", type=Path, help="new baseline destination (requires --tensor-hashes)")
    ap.add_argument("--hf", metavar="USER/REPO", help="cross-check local sha256 against HF LFS hash")
    ap.add_argument("--revision", metavar="REVISION", default=None, help="HF revision to resolve; output records the full resolved commit")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--diff", metavar=("A", "B"), nargs=2, default=None, help="compare two .tensorhashes.json baselines")
    a = ap.parse_args()
    if a.baseline_out is not None and (not a.tensor_hashes or a.diff):
        ap.error("--baseline-out requires --tensor-hashes and an audit, not --diff")
    if a.diff:
        return run_cli_report(lambda rep: diff_baselines(a.diff[0], a.diff[1], rep), a.json, "baseline")
    if not a.model:
        ap.print_help()
        return 0
    return run_cli_report(lambda rep: audit(a.model, rep, do_hashes=a.tensor_hashes,
        hf_repo=a.hf, hf_revision=a.revision, baseline_out=a.baseline_out), a.json, "structure")


if __name__ == "__main__":
    raise SystemExit(main())
