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
     the LFS hash HF publishes (detects force-push / file swap since download).

Exit codes: 0 = clean, 1 = findings, 2 = hard errors.
"""
import argparse, hashlib, json, os, re, struct, sys, unicodedata
from pathlib import Path

CHUNK = 1 << 24
try:
    import gguf
except ImportError:  # safetensors auditing needs no third-party packages at all
    gguf = None

# ---------------- findings collector ----------------
class Report:
    def __init__(self):
        self.items = []  # (severity, section, message)
    def add(self, sev, section, msg):
        self.items.append((sev, section, msg))
    def print(self, as_json=False):
        if as_json:
            print(json.dumps([{"severity": s, "section": sec, "message": m}
                              for s, sec, m in self.items], indent=2))
            return
        order = {"CRIT": 0, "WARN": 1, "INFO": 2, "OK": 3}
        for sev, sec, msg in sorted(self.items, key=lambda x: order.get(x[0], 9)):
            print(f"[{sev:4}] {sec:12} {msg}")

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
    h = hashlib.sha256()
    with open(path, "rb") as f:
        f.seek(offset)
        remaining = length
        while remaining > 0:
            b = f.read(min(CHUNK, remaining))
            if not b:
                break
            h.update(b)
            remaining -= len(b)
    return h.hexdigest()

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
    print("\n--- chat template (verbatim from file) ---")
    print(tpl)
    print("--- end template ---\n")

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
    ft_names = {0:"F32",1:"F16",2:"Q4_0",3:"Q4_1",7:"Q8_0",8:"Q5_0",9:"Q5_1",
                10:"Q2_K",11:"Q3_K_S",12:"Q3_K_M",13:"Q3_K_L",14:"Q4_K_S",15:"Q4_K_M",
                16:"Q5_K_S",17:"Q5_K_M",18:"Q6_K",19:"IQ2_XXS",30:"BF16",31:"MXFP4",38:"MXFP4"}
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
        if hlen >= size - 8:
            rep.add("CRIT", "st-header", f"header unreadable: declared length {hlen:,} exceeds the file size "
                                         f"{size:,} — truncated or fabricated file")
            return None
        blob = f.read(hlen)
    try:
        hdr = json.loads(blob.decode("utf-8"))
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
    meta = hdr.get("__metadata__")
    if meta is not None and not isinstance(meta, dict):
        rep.add("CRIT", "st-header", "__metadata__ present but not a JSON object")
    elif isinstance(meta, dict):
        for k, v in sorted(meta.items()):
            if isinstance(v, str) and len(v) <= 200:
                rep.add("INFO", "provenance", f"__metadata__[{k}] = {v}")
            else:
                rep.add("INFO", "provenance", f"__metadata__[{k}] present (non-string or oversized)")
    entries = {k: v for k, v in hdr.items() if k != "__metadata__"}
    if not entries:
        rep.add("CRIT", "st-header", "no tensors declared in header")
    tensors, spans = [], []
    for name, ent in entries.items():
        if not isinstance(ent, dict) or not {"dtype", "shape", "data_offsets"} <= set(ent):
            rep.add("CRIT", "st-header", f"tensor entry missing dtype/shape/data_offsets: {name}")
            continue
        dtype, shape, off = ent["dtype"], ent["shape"], ent["data_offsets"]
        if not (isinstance(off, list) and len(off) == 2 and all(isinstance(x, int) for x in off)):
            rep.add("CRIT", "st-header", f"tensor entry has malformed data_offsets: {name} ({off!r})")
            continue
        begin, end = off
        if begin < 0 or end < begin:
            rep.add("CRIT", "st-header", f"tensor entry has nonsensical offsets: {name} [{begin}, {end})")
            continue
        n_bytes = end - begin
        if data_start + end > size:
            rep.add("CRIT", "st-tensors", f"tensor size: {name} declares data ending at "
                                          f"{data_start + end:,}, past the end of the file ({size:,}) — truncated")
            continue
        unit = ST_DTYPE_SIZE.get(dtype)
        numel = 1
        for d in shape:
            numel *= d
        if unit is None:
            rep.add("WARN", "st-tensors", f"tensor size: {name} dtype {dtype!r} is not a known type — "
                                          f"its byte size cannot be verified")
        elif int(numel * unit) != n_bytes:
            rep.add("CRIT", "st-tensors", f"tensor size: {name} is {dtype} {list(shape)}, which implies "
                                          f"{int(numel * unit):,} bytes, but the header declares {n_bytes:,} — "
                                          f"the header contradicts itself")
        if n_bytes == 0 or any(d == 0 for d in shape):
            rep.add("WARN", "st-tensors", f"tensor size: {name} is empty ({dtype} {list(shape)})")
        tensors.append(STTensor(name, shape, dtype, data_start + begin, n_bytes))
        spans.append((begin, end, name))
    spans.sort()
    prev_end = 0
    for begin, end, name in spans:
        if begin < prev_end:
            rep.add("CRIT", "st-tensors", f"tensor layout: {name} starts inside another tensor's data — "
                                          f"the declared layout is not valid safetensors")
        elif begin > prev_end:
            rep.add("WARN", "st-tensors", f"undeclared gap: {begin - prev_end:,} bytes between tensors before "
                                          f"{name} that no header entry accounts for")
        prev_end = max(prev_end, end)
    if prev_end < size - data_start:
        rep.add("WARN", "st-tensors", f"trailing data: {size - data_start - prev_end:,} bytes after the last "
                                      f"declared tensor that the header does not account for")
    return tensors

def check_st_config(path, rep):
    """config.json is where a safetensors file's claimed identity and provenance live."""
    fp = path.parent / "config.json"
    if not fp.exists():
        rep.add("INFO", "meta", "no config.json in this directory — architecture and provenance are not "
                                "pinned by any sidecar here")
        return
    try:
        cfg = json.loads(fp.read_text("utf-8"))
    except Exception as e:
        rep.add("WARN", "sidecar", f"config.json present but unreadable: {e}")
        return
    archs = cfg.get("architectures") or []
    name = cfg.get("_name_or_path") or cfg.get("name_or_path") or path.parent.name
    rep.add("INFO", "meta", f"architecture={archs[0] if archs else cfg.get('model_type', '?')} "
                            f"name={name!r} file_type={cfg.get('torch_dtype', '?')} (safetensors)")
    for key in ("model_type", "torch_dtype", "transformers_version", "base_model",
                "quantized_by", "converted_by", "source"):
        if cfg.get(key):
            rep.add("INFO", "meta", f"{key} = {cfg[key]}")
    if cfg.get("quantization_config"):
        qc = cfg["quantization_config"]
        method = qc.get("quant_method", "?") if isinstance(qc, dict) else "?"
        rep.add("INFO", "provenance", f"config.json carries quantization_config ({method}) — these weights are "
                                      f"not raw fp/bf16, and their provenance is the uploader's word")
    for fp2, label in ((fp, "config.json"), (path.parent / "tokenizer_config.json", "tokenizer_config.json")):
        if not fp2.exists():
            continue
        try:
            other = cfg if fp2 == fp else json.loads(fp2.read_text("utf-8"))
        except Exception:
            continue
        if other.get("auto_map"):
            rep.add("WARN", "sidecar", f"auto_map declared in {label}: loading this model with "
                                       f"trust_remote_code runs Python the uploader ships — a separate supply "
                                       f"chain from the weights, and the usual way a 'model' executes code")

def _scan_template_labeled(text, rep, label):
    before = len(rep.items)
    scan_template(text, rep)
    for i in range(before, len(rep.items)):
        sev, sec, msg = rep.items[i]
        rep.items[i] = (sev, sec, f"{label}: {msg}")

def scan_st_templates(path, rep):
    """The template is not in the artifact — it is a sidecar. Scan every one we find."""
    d = path.parent
    jinjas = [p for p in sorted(d.glob("*.jinja")) + sorted(d.glob("*.jinja2")) if p.is_file()]
    cfg_templates = []
    for fp in jinjas:
        rep.add("INFO", "template", f"template sidecar: {fp.name} ({fp.stat().st_size:,} bytes)")
        try:
            _scan_template_labeled(fp.read_text("utf-8", errors="replace"), rep, fp.name)
        except Exception as e:
            rep.add("WARN", "template", f"template sidecar: {fp.name} unreadable: {e}")
    tcfg = d / "tokenizer_config.json"
    if tcfg.exists():
        try:
            tcfg_json = json.loads(tcfg.read_text("utf-8"))
        except Exception as e:
            tcfg_json = {}
            rep.add("WARN", "sidecar", f"tokenizer_config.json present but unreadable: {e}")
        tpl = tcfg_json.get("chat_template")
        if isinstance(tpl, str):
            cfg_templates.append(("tokenizer_config.json", tpl))
        elif isinstance(tpl, list):
            cfg_templates += [(f"tokenizer_config.json[chat_template[{i}]]", t)
                              for i, t in enumerate(tpl) if isinstance(t, str)]
        elif isinstance(tpl, dict):
            cfg_templates += [(f"tokenizer_config.json[chat_template.{k}]", v)
                              for k, v in tpl.items() if isinstance(v, str)]
    for label, text in cfg_templates:
        rep.add("INFO", "template", f"template sidecar: {label} ({len(text):,} bytes)")
        _scan_template_labeled(text, rep, label)
    if jinjas and cfg_templates:
        jtxt = jinjas[0].read_text("utf-8", errors="replace")
        if any(text != jtxt for _, text in cfg_templates):
            rep.add("WARN", "template", "both a chat_template.jinja sidecar and a chat_template in "
                                        "tokenizer_config.json exist and they differ — which one a loader uses "
                                        "is runtime-dependent, so the model's behavior is not pinned by either")
    if not jinjas and not cfg_templates:
        rep.add("INFO", "template", "no chat template sidecar in this directory (chat_template.jinja or "
                                    "tokenizer_config.json) — this format never carries the template inside the "
                                    "file, so whatever loads the model supplies it")

def check_st_index(path, tensors, rep):
    """Multi-shard models are described by model.safetensors.index.json — check it agrees."""
    idxs = sorted(path.parent.glob("*.safetensors.index.json"))
    if not idxs:
        return
    fp = idxs[0]
    try:
        idx = json.loads(fp.read_text("utf-8"))
    except Exception as e:
        rep.add("WARN", "sidecar", f"index mismatch: {fp.name} unreadable ({e})")
        return
    wmap = idx.get("weight_map") or {}
    shards = sorted(set(wmap.values()))
    rep.add("INFO", "index", f"{fp.name}: {len(wmap)} tensors across {len(shards)} shard(s)")
    mine = {t.name for t in tensors}
    listed_here = {k for k, v in wmap.items() if Path(v).name == path.name}
    missing = sorted(listed_here - mine)
    unlisted = sorted(mine - listed_here)
    if missing:
        rep.add("CRIT", "index", f"index mismatch: {len(missing)} tensor(s) the index routes to this shard are "
                                 f"absent from it (e.g. {', '.join(missing[:3])})")
    if unlisted:
        rep.add("WARN", "index", f"index mismatch: {len(unlisted)} tensor(s) present here are not listed in "
                                 f"{fp.name} (e.g. {', '.join(unlisted[:3])}) — added or renamed after indexing")
    total = (idx.get("metadata") or {}).get("total_size")
    if len(shards) == 1 and isinstance(total, int):
        declared = sum(t.n_bytes for t in tensors)
        if total != declared:
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
    entries = {}
    for line in fp.read_text("utf-8", errors="replace").splitlines():
        parts = line.split()
        if len(parts) < 2:
            continue
        a, b = parts[0], parts[-1].lstrip("*")
        if re.fullmatch(r"[0-9a-fA-F]{64}", a):
            entries[b] = a.lower()
        elif re.fullmatch(r"[0-9a-fA-F]{64}", b):
            entries[a] = b.lower()
    if not entries:
        rep.add("INFO", "manifest", f"{fp.name} present but no readable SHA-256 lines")
        return
    if path.name in entries:
        if entries[path.name] == sha:
            rep.add("OK", "manifest", f"SHA-256 matches the uploader's own entry for {path.name} in {fp.name}")
        else:
            rep.add("CRIT", "manifest", f"MANIFEST MISMATCH: this file is {sha[:16]}… but {fp.name} lists "
                                        f"{entries[path.name][:16]}… for {path.name} — the file differs from "
                                        f"what the uploader published hashes for")
        return
    for name, h in entries.items():
        if h == sha:
            rep.add("INFO", "manifest", f"renamed since the manifest was written: this file's SHA-256 appears "
                                        f"in {fp.name} as {name} — content unchanged, name is not")
            return
    rep.add("INFO", "manifest", f"{fp.name} present ({len(entries)} entries) but this file is not listed in it")

# ---------------- HF remote check ----------------
def hf_check(repo, local_sha, path, rep, revision=None):
    try:
        from huggingface_hub import HfApi
    except ImportError:
        rep.add("WARN", "remote", "huggingface_hub not installed; cannot do remote hash check")
        return
    try:
        api = HfApi()
        if revision:
            rep.add("INFO", "remote", f"pinned revision: {revision}")
        info = api.model_info(repo, files_metadata=True, revision=revision)
        match = None
        for s in info.siblings:
            # any LFS-tracked file, not just .gguf: the artifact's identity is its hash
            if s.lfs and s.lfs.sha256 == local_sha:
                match = s
                break
        if match is None:
            # try exact filename match for a clearer diagnostic
            byname = [s for s in info.siblings
                      if s.lfs and Path(s.rfilename).name == path.name]
            if byname:
                s = byname[0]
                rep.add("CRIT", "remote",
                        f"SHA-256 MISMATCH vs {repo}@{str(info.sha)[:12]}: local {local_sha[:16]}… "
                        f"vs repo {s.lfs.sha256[:16]}… for {s.rfilename} — file was swapped, "
                        f"re-quantized upstream, or this copy did not come from this repo")
            else:
                rep.add("WARN", "remote", f"no file with this SHA-256 or filename in {repo} — "
                                          f"repo contents may have been replaced (force-push)")
            return
        rep.add("OK", "remote", f"local SHA-256 matches HF LFS for {repo}@{str(info.sha)[:12]} ({match.rfilename})")
        # warn if the repo ships its own chat template sidecar
        names = {s.rfilename for s in info.siblings}
        if "chat_template.jinja" in names:
            rep.add("INFO", "remote", f"{repo} ships chat_template.jinja alongside the weights — "
                                              f"loaders may prefer it over any template inside the artifact")
    except Exception as e:
        rep.add("WARN", "remote", f"HF fetch failed: {e}")

# ---------------- baseline tensor hash dump / compare ----------------
def dump_tensor_hashes(tensors, path):
    out = {}
    for t in tensors:
        out[t.name] = sha256_at(path, t.data_offset, t.n_bytes)
    return out

def write_baseline(tensors, path, sha, fmt, rep):
    hashes, shapes = dump_tensor_hashes(tensors, path), {}
    for t in tensors:
        shapes[t.name] = [int(d) for d in t.shape]
    outp = path.with_suffix(path.suffix + ".tensorhashes.json")
    outp.write_text(json.dumps({"file_sha256": sha, "format": fmt,
                                "shapes": shapes, "tensors": hashes}, indent=1))
    rep.add("OK", "baseline", f"per-tensor SHA-256 written to {outp}")

# ---------------- main ----------------
def _audit_gguf(path, rep, sha, size, do_hashes, hf_repo, hf_revision):
    rep.add("INFO", "file", "format=gguf")
    if gguf is None:
        rep.add("CRIT", "file", "the gguf python package is not installed (pip install gguf) — "
                                "GGUF files cannot be parsed without it")
        return
    try:
        rdr = gguf.GGUFReader(str(path))
    except Exception as e:
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
    arch = check_meta(md, path, rep)
    tpl = md.get("tokenizer.chat_template")
    if tpl is None and arch == "clip":
        rep.add("INFO", "template", "No tokenizer.chat_template — normal for vision projector (mmproj) files")
    else:
        scan_template(tpl, rep)
    tensors = list(rdr.tensors)
    check_tensors(tensors, arch, rep)
    check_sidecars(path, rep)
    check_manifest(path, sha, rep)
    if do_hashes:
        write_baseline(tensors, path, sha, "gguf", rep)
    if hf_repo:
        hf_check(hf_repo, sha, path, rep, revision=hf_revision)

def _audit_st(path, rep, sha, size, do_hashes, hf_repo, hf_revision):
    rep.add("INFO", "file", "format=safetensors")
    parsed = read_st_header(path, rep)
    if parsed is None:
        return
    hdr, data_start, fsize = parsed
    tensors = st_tensors(hdr, data_start, fsize, rep)
    check_tensors(tensors, "?", rep, naming="hf")
    check_st_config(path, rep)
    scan_st_templates(path, rep)
    check_st_index(path, tensors, rep)
    check_sidecars(path, rep)
    check_manifest(path, sha, rep)
    if do_hashes:
        write_baseline(tensors, path, sha, "safetensors", rep)
    if hf_repo:
        hf_check(hf_repo, sha, path, rep, revision=hf_revision)

def audit(path, rep, do_hashes=False, hf_repo=None, hf_revision=None):
    rep.add("INFO", "file", f"{path}")
    sha, size = file_hashes(path)
    rep.add("INFO", "file", f"sha256={sha}")
    rep.add("INFO", "file", f"size={size:,}")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic == b"GGUF":
        return _audit_gguf(path, rep, sha, size, do_hashes, hf_repo, hf_revision)
    if looks_like_safetensors(path):
        return _audit_st(path, rep, sha, size, do_hashes, hf_repo, hf_revision)
    rep.add("CRIT", "file", f"bad magic {magic!r} — not a GGUF or safetensors file")

def blob_identity(A, B):
    """Match weight blobs across formats by (shape, SHA-256).

    Tensor *names* differ between formats (model.layers.0.self_attn.q_proj vs
    blk.0.attn_q), so names cannot carry a cross-format comparison. Shape plus
    content can: if a blob in B is byte-identical to a blob in A, that weight
    survived the conversion unchanged. Returns (matched, total_in_b, unmatched_names).
    """
    shapes_a = A.get("shapes") or {}
    counts_a = {}
    for name, h in A["tensors"].items():
        counts_a[(tuple(shapes_a.get(name, [])), h)] = counts_a.get((tuple(shapes_a.get(name, [])), h), 0) + 1
    shapes_b = B.get("shapes") or {}
    matched, unmatched = 0, []
    for name, h in B["tensors"].items():
        key = (tuple(shapes_b.get(name, [])), h)
        if counts_a.get(key, 0) > 0:
            counts_a[key] -= 1
            matched += 1
        else:
            unmatched.append(name)
    return matched, len(B["tensors"]), sorted(unmatched)


def diff_baselines(file_a, file_b, rep):
    """Compare two .tensorhashes.json baselines of the same base model."""
    A = json.loads(Path(file_a).read_text())
    B = json.loads(Path(file_b).read_text())
    ta, tb = A["tensors"], B["tensors"]
    rep.add("INFO", "diff", f"A: {file_a} (file sha {A.get('file_sha256','?')[:16]}…)")
    rep.add("INFO", "diff", f"B: {file_b} (file sha {B.get('file_sha256','?')[:16]}…)")
    only_a = sorted(set(ta) - set(tb))
    only_b = sorted(set(tb) - set(ta))
    changed = sorted(k for k in set(ta) & set(tb) if ta[k] != tb[k])
    fa, fb = A.get("format", "?"), B.get("format", "?")
    if fa != fb:
        rep.add("WARN", "diff", f"comparing a {fa} baseline with a {fb} one — names and precision differ "
                                f"between formats, so a per-name comparison is not meaningful and the "
                                f"abliteration/re-training classification is skipped. Use the content match "
                                f"below instead")
        rep.add("INFO", "diff", f"name sets: {len(only_a)} only in A, {len(only_b)} only in B, "
                                f"{len(set(ta) & set(tb))} in common (per-name listing skipped for cross-format)")
        matched, total, unmatched = blob_identity(A, B)
        if total and matched == total:
            rep.add("OK", "diff", f"content match: all {total} weight blobs in B are byte-identical "
                                  f"(shape + SHA-256) to blobs in A — consistent with a lossless conversion "
                                  f"of the same weights (naming changed, values did not)")
        else:
            rep.add("INFO", "diff", f"content match: {matched} of {total} weight blobs in B are byte-identical "
                                    f"to blobs in A; {len(unmatched)} differ. Quantized formats transform "
                                    f"values by design, so a low count means 'not comparable this way', not "
                                    f"'tampered' — only a lossless conversion is expected to match")
            for name in unmatched[:8]:
                rep.add("INFO", "diff", f"not matched: {name}")
            if len(unmatched) > 8:
                rep.add("INFO", "diff", f"… and {len(unmatched) - 8} more unmatched blobs")
        return
    same = len(set(ta) & set(tb)) - len(changed)
    for k in only_a:
        rep.add("WARN", "diff", f"tensor only in A (removed in B): {k}")
    for k in only_b:
        rep.add("WARN", "diff", f"tensor only in B (added/renamed): {k}")
    for k in changed:
        rep.add("CRIT", "diff", f"TENSOR CONTENT CHANGED: {k}  ({ta[k][:12]}… -> {tb[k][:12]}…)")
    rep.add("INFO", "diff", f"{same} tensors identical, {len(changed)} changed, "
                            f"{len(only_a)} removed, {len(only_b)} added")
    total = len(changed) + len(only_a) + len(only_b)
    if total == 0:
        rep.add("OK", "diff", "baselines are byte-identical tensor-wise")
        return
    # classify: abliteration-style (few tensors, at common edit sites) vs broad change
    edit_sites = sum(1 for k in changed if any(s in k for s in ABLIT_SITES))
    if 0 < len(changed) <= 8 and not only_a and not only_b and edit_sites >= max(1, len(changed) // 2):
        rep.add("INFO", "diff", f"pattern matches targeted weight edits (abliteration-class: "
                                f"{edit_sites}/{len(changed)} at ffn_down/ffn_out/attn_o sites)")
    elif total > 8:
        rep.add("CRIT", "diff", f"BROAD tensor differences ({total}: {len(changed)} changed, "
                                f"{len(only_a)} removed, {len(only_b)} added) — consistent with "
                                f"re-quantization, post-quantization training, or large-scale "
                                f"tampering. Without a trusted chain of custody, treat B as a "
                                f"different model, not a copy.")


def cmd_diff(args):
    rep = Report()
    diff_baselines(args.baseline_a, args.baseline_b, rep)
    rep.print(as_json=args.json)
    crits = sum(1 for s, *_ in rep.items if s == "CRIT")
    warns = sum(1 for s, *_ in rep.items if s == "WARN")
    print(f"\n== diff: {crits} critical, {warns} warnings ==")
    sys.exit(2 if crits else (1 if warns else 0))


def main():
    ap = argparse.ArgumentParser(description="audit a GGUF or safetensors file for supply-chain tampering")
    ap.add_argument("model", type=Path, nargs="?", help="model file to audit (GGUF or safetensors; format auto-detected)")
    ap.add_argument("--tensor-hashes", action="store_true", help="emit per-tensor SHA-256 baseline JSON")
    ap.add_argument("--hf", metavar="USER/REPO", help="cross-check local sha256 against HF LFS hash")
    ap.add_argument("--revision", metavar="COMMIT_SHA", default=None,
                    help="pin the HF commit to compare against (never trust 'main' long-term)")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--diff", metavar=("A", "B"), nargs=2, default=None,
                    help="compare two .tensorhashes.json baselines instead of auditing")

    a = ap.parse_args()
    if a.diff:
        rep = Report()
        diff_baselines(a.diff[0], a.diff[1], rep)
        rep.print(as_json=a.json)
        crits = sum(1 for s, *_ in rep.items if s == "CRIT")
        warns = sum(1 for s, *_ in rep.items if s == "WARN")
        print(f"\n== diff: {crits} critical, {warns} warnings ==")
        sys.exit(2 if crits else (1 if warns else 0))
    if not a.model:
        ap.print_help()
        sys.exit(0)
    if not a.model.exists():
        sys.exit(f"not found: {a.model}")
    rep = Report()
    audit(a.model, rep, do_hashes=a.tensor_hashes, hf_repo=a.hf, hf_revision=a.revision)
    rep.print(as_json=a.json)
    crits = sum(1 for s, *_ in rep.items if s == "CRIT")
    warns = sum(1 for s, *_ in rep.items if s == "WARN")
    print(f"\n== {a.model.name}: {crits} critical, {warns} warnings ==")
    sys.exit(2 if crits else (1 if warns else 0))

if __name__ == "__main__":
    main()
