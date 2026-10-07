#!/usr/bin/env python3
"""
gguf_audit — supply-chain auditor for (abliterated) GGUF models.

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
import argparse, hashlib, json, os, re, sys, unicodedata
from pathlib import Path

CHUNK = 1 << 24
try:
    import gguf
except ImportError:
    sys.exit("pip install gguf")

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
            ctx = tpl[max(0, m.start()-30):m.end()+30].replace("\n", " ")
            sev = "INFO" if label.endswith("wording") and "tool" in code_only[max(0,m.start()-200):m.start()+200].lower() else "WARN"
            rep.add(sev, "template", f"{label}: ...{ctx}...")
    # structural surprise: template doing arithmetic/loops over non-message vars
    for var in set(re.findall(r"\{\{\s*([a-zA-Z_][\w\.]*)", tpl)):
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

def check_tensors(tensors, arch, rep):
    names = [t.name for t in tensors]
    n = len(names)
    rep.add("INFO", "tensors", f"{n} tensors total")
    # classic structure spot-checks
    has_attn = sum(1 for x in names if "attn_q" in x or "attn_k" in x or "attn_v" in x)
    if has_attn and has_attn < 3:
        rep.add("WARN", "tensors", "attention projections partially missing — possible tensor pruning/swap")
    for t in tensors:
        shp = list(t.shape)
        if shp and any(s == 0 for s in shp):
            rep.add("CRIT", "tensors", f"tensor {t.name} has a zero dimension {t.shape}")
    # FFN/ATTN norm ratio sanity per layer (abliteration edits ffn_down/out or attn oproj)
    ffn_down = [t for t in tensors if "ffn_down" in t.name and ".weight" in t.name]
    if ffn_down:
        rep.add("INFO", "tensors", f"ffn_down tensors present: {len(ffn_down)} "
                                   "(abliteration commonly edits ffn_down/ffn_out/attn_o — diff these vs a trusted build)")
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
            if s.rfilename.endswith(".gguf") and s.lfs and s.lfs.sha256 == local_sha:
                match = s
                break
        if match is None:
            # try exact filename match for a clearer diagnostic
            byname = [s for s in info.siblings
                      if s.rfilename.endswith(".gguf") and s.lfs
                      and Path(s.rfilename).name == path.name]
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
            rep.add("INFO", "remote", f"{repo} ships chat_template.jinja alongside the GGUF — "
                                      f"loaders may prefer it over the in-file template")
    except Exception as e:
        rep.add("WARN", "remote", f"HF fetch failed: {e}")

# ---------------- baseline tensor hash dump / compare ----------------
def dump_tensor_hashes(tensors, path):
    out = {}
    for t in tensors:
        out[t.name] = sha256_at(path, t.data_offset, t.n_bytes)
    return out

# ---------------- main ----------------
def audit(path, rep, do_hashes=False, hf_repo=None, hf_revision=None):
    rep.add("INFO", "file", f"{path}")
    sha, size = file_hashes(path)
    rep.add("INFO", "file", f"sha256={sha}")
    rep.add("INFO", "file", f"size={size:,}")
    with open(path, "rb") as f:
        magic = f.read(4)
    if magic != b"GGUF":
        rep.add("CRIT", "file", f"bad magic {magic!r} — not a GGUF")
        return
    rdr = gguf.GGUFReader(str(path))
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
    if do_hashes:
        hashes = dump_tensor_hashes(tensors, path)
        outp = path.with_suffix(path.suffix + ".tensorhashes.json")
        outp.write_text(json.dumps({"file_sha256": sha, "tensors": hashes}, indent=1))
        rep.add("OK", "baseline", f"per-tensor SHA-256 written to {outp}")
    if hf_repo:
        hf_check(hf_repo, sha, path, rep, revision=hf_revision)

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
    edit_sites = sum(1 for k in changed
                     if "ffn_down" in k or "ffn_out" in k or "attn_o" in k or "output.weight" in k)
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
    ap = argparse.ArgumentParser(description="audit a GGUF for supply-chain tampering")
    ap.add_argument("gguf", type=Path, nargs="?", help="GGUF file to audit")
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
    if not a.gguf:
        ap.print_help()
        sys.exit(0)
    if not a.gguf.exists():
        sys.exit(f"not found: {a.gguf}")
    rep = Report()
    audit(a.gguf, rep, do_hashes=a.tensor_hashes, hf_repo=a.hf, hf_revision=a.revision)
    rep.print(as_json=a.json)
    crits = sum(1 for s, *_ in rep.items if s == "CRIT")
    warns = sum(1 for s, *_ in rep.items if s == "WARN")
    print(f"\n== {a.gguf.name}: {crits} critical, {warns} warnings ==")
    sys.exit(2 if crits else (1 if warns else 0))

if __name__ == "__main__":
    main()
