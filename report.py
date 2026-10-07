"""
report.py — human-friendly HTML report generator for model_audit results.

Point it at one or more model files (GGUF or safetensors); it runs the full audit
and writes a plain-language HTML report (plus machine-readable JSON) to reports/.

Usage:
    python3 report.py <file1> [file2 ...] [-o reports/]
"""
import argparse, json, sys, html
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from model_audit import Report, audit as _audit

# plain-language translations for every finding we emit
PLAIN = {
    "No tokenizer.chat_template": (
        "This file has NO chat template inside it. The template is what tells the model "
        "how to hold a conversation — without one baked in, whatever app you load it with "
        "substitutes its own or whatever is in a sidecar config file. You can't know from "
        "the file alone how it will actually behave."),
    "no tokenizer.ggml.model": (
        "The tokenizer (the piece that converts text to numbers) is not described in this "
        "file's metadata — normal for vision attachments (mmproj), unexpected for a main model file."),
    "jinja import": "The template can import Python code — a legitimate template never needs this.",
    "python dunder traversal": "The template reaches into Python's internals (__class__, __subclasses__...). "
        "This is the classic escape hatch used to run arbitrary code. A normal template never does this.",
    "dynamic attribute access": "The template can fetch arbitrary attributes by name — used together with "
        "other tricks to escape the safe template sandbox.",
    "file I/O wording": "The template mentions file reading/writing. Can be a false positive in models with "
        "tool-calling (they legitimately talk about 'files' as conversation topics), but worth reading the context.",
    "exec/OS wording": "The template mentions running programs or OS commands. Like the above: often legitimate "
        "tool-calling vocabulary, but the surrounding text should make sense.",
    "environment/config access": "The template reaches for environment or configuration data.",
    "suspicious namespace() usage": "The template abuses a template feature in a way that references system "
        "objects — not a pattern honest templates use.",
    "zero-width": "Invisible characters are hidden in the template. These can smuggle instructions past a human "
        "reviewer or alter model behavior without any visible change.",
    "confusable": "The template contains look-alike characters (e.g. Cyrillic 'а' inside English text). "
        "Sometimes innocent, sometimes used to hide trickery from review.",
    "attention projections partially missing": "Some expected model pieces appear to be absent — could be a "
        "corrupt or deliberately modified build.",
    "zero dimension": "A tensor (weight block) in this model is empty — the file is likely broken or was tampered with.",
    "filename says": "The filename claims one quantization level but the file's internal metadata says another. "
        "At best sloppy labeling; at worst misrepresentation.",
    "no general.architecture": "The file doesn't even say what kind of model it is — nonstandard or stripped build.",
    "executable/script shipped alongside": "A program/script ships in the same folder as the model. The model "
        "weights are safe data — anything executable next to them is a separate supply-chain risk. Don't run it.",
    "SHA-256 MISMATCH": "The file's fingerprint does NOT match what the Hugging Face repo currently publishes — "
        "it was swapped, re-uploaded, or this copy didn't come from the repo.",
    "MANIFEST MISMATCH": "The uploader published a list of hashes with this model, and this file does not match "
        "the entry under its own name. Either the file was changed after it was hashed, or it was replaced. "
        "Treat that file as untrustworthy until the uploader explains it.",
    "renamed since the manifest was written": "The uploader's own hash list still has this file under its old "
        "name, and the content matches. That is a rename without a content change — harmless in itself, and it "
        "shows the hash list is still accurate.",
    "present but no readable SHA-256 lines": "A checksum file ships with this model but this tool could not "
        "read any hashes out of it.",
    "but this file is not listed in it": "A checksum file ships with this model but does not cover this file, "
        "so it can say nothing about it either way.",
    "local SHA-256 matches HF LFS": "The file's fingerprint matches what Hugging Face publishes right now.",
    "per-tensor SHA-256 written": "A per-weight-block fingerprint baseline was saved for future comparison.",
    "config file present": "A config file sits next to the model; its chat settings could differ from what's "
        "inside the GGUF. If they disagree, trust the GGUF.",
    "references non-standard variable": "The template uses a variable that isn't part of the standard chat "
        "vocabulary — usually fine (tool-calling), occasionally interesting.",
    "references non-standard variable: messages_json": "",
    "tensors sit at common abliteration edit sites": "Weight-block census recorded — these are the layers "
        "abliteration edits, so they are what to diff against a trusted build of the same base.",
    "no GGUF named": "The repo doesn't contain a file by this name anymore — it may have been replaced.",
    # --- safetensors ---
    "header unreadable": "The file's own header is broken: it claims a size that cannot be true for this "
        "file. That is either a corrupt download or a fabricated file — nothing about it can be trusted.",
    "header invalid": "The header is not the JSON structure safetensors requires. This file is not a valid "
        "safetensors file, whatever its name says.",
    "no tensors declared": "The header describes no weights at all — an empty or stripped file.",
    "tensor entry missing dtype/shape/data_offsets": "A weight entry in the header is missing the fields "
        "that describe it. The file cannot be read reliably.",
    "tensor entry has malformed data_offsets": "A weight entry points at its data with invalid numbers — "
        "the file's internal map is broken.",
    "tensor entry has nonsensical offsets": "A weight entry declares an impossible data range (it ends "
        "before it starts, or starts before zero).",
    "tensor size:": "A weight block's declared size does not match what its type and shape require. The "
        "header contradicts itself, which is what a hand-edited or grafted file looks like.",
    "tensor layout:": "Two weight blocks claim the same bytes — the file's layout is not a valid "
        "safetensors layout.",
    "undeclared gap": "There is a stretch of data inside the file that no header entry claims. It could be "
        "harmless padding from an unusual writer, or a region hidden from any structural review.",
    "trailing data": "There are bytes after the last declared weight. Nothing in the header accounts for "
        "them: appended data is invisible to anyone who only checks the weights.",
    "is empty": "A weight block is empty — the file is likely broken or was modified.",
    "dtype is not a known type": "A weight declares a data type this tool does not recognize, so its size "
        "cannot be verified.",
    "no config.json in this directory": "There is no config beside this file. The model's claimed identity "
        "and provenance live in sidecar files, and none are here — treat the file as unnamed.",
    "config.json present but unreadable": "A config file is here but cannot be parsed.",
    "auto_map": "The config tells loaders to run Python code shipped by the uploader. If you load this "
        "model with 'trust remote code' enabled, that code runs on your machine — a completely separate "
        "supply chain from the weights. Only proceed if you trust the uploader and have read that code.",
    "quantization_config": "The config says these weights are quantized rather than raw. That is fine, but "
        "the claim comes from the uploader — nothing here verifies how they were made.",
    "index mismatch": "The index file that describes a multi-shard model disagrees with this shard: it "
        "lists weights that are not here, or misses weights that are. Shards and index must match exactly.",
    "template sidecar": "This template file sits next to the weights and is what actually shapes the "
        "conversation. A slow, careful review of this one small file covers the same attack surface as "
        "reviewing the whole model.",
    "no chat template sidecar": "This format never carries the chat template inside the file, and no "
        "template sidecar is present. Whatever you load the model with supplies the template, so its "
        "behavior is not pinned by anything you audited here.",
    "both a chat_template.jinja": "Two different templates ship in the same folder. Which one a given app "
        "uses is a runtime detail, so you cannot tell from the files alone how the model will behave.",
    "__metadata__": "Free-form notes the uploader put in the header. They are self-reported and prove "
        "nothing, but they often name the tool and the source the file came from.",
    "the gguf python package is not installed": "GGUF files need a small helper library that is not "
        "installed here, so this file could not be inspected at all.",
    "GGUF header unreadable": "This file claims to be a GGUF but its header cannot be parsed — it is "
        "corrupt, truncated, or not really a GGUF.",
    "bad magic": "The file's first bytes match no known model format — it is not a GGUF or safetensors "
        "file, whatever its name says.",
    "comparing a": "These two fingerprints were made from different formats (for example a GGUF against a "
        "safetensors file). Names and precision differ between formats, so the differences listed below are "
        "expected and the verdict is not meaningful.",
}

SEV_COLOR = {"CRIT": "#c0392b", "WARN": "#d68910", "INFO": "#5d6d7e", "OK": "#1e8449"}
SEV_LABEL = {"CRIT": "CRITICAL", "WARN": "WARNING", "INFO": "Info", "OK": "OK"}

def plain_for(sev, sec, msg):
    for k, v in PLAIN.items():
        if msg.startswith(k) or k in msg[:40] or k in msg:
            return v
    return ""

def verdict(crits, warns):
    if crits:
        return ("DO NOT RUN THIS FILE", SEV_COLOR["CRIT"],
                "Critical findings mean this file is unsafe or untrustworthy as-is.",
                "What to do: delete or quarantine this file. If you want this model, get it from "
                "a source that documents its quantization (or quantize it yourself), then re-audit.")
    if warns:
        return ("USE WITH AWARENESS", SEV_COLOR["WARN"],
                "Warnings mean something needs your attention, but the file isn't clearly hostile. Read the details below.",
                "What to do: read each warning's explanation below — some are false positives from "
                "tool-calling models. If the model matters to you, run <code>python3 model_audit.py "
                "&lt;file&gt; --tensor-hashes</code> to save a per-weight fingerprint baseline.")
    return ("NO RED FLAGS FOUND", SEV_COLOR["OK"],
            "Nothing hostile was detected. Note: this audit can't prove weights are 'good' — "
            "only that nothing obviously bad was found in metadata and the chat template. "
            "Per-weight fingerprints (tensor hashes) are needed to compare against a trusted build.",
            "What to do: if this model matters to you, save a fingerprint baseline now: "
            "<code>python3 model_audit.py &lt;file&gt; --tensor-hashes</code>. Keep the JSON next "
            "to the model — it lets you prove later that the weights haven't changed.")

def render_file_block(name, items):
    crits = sum(1 for s, *_ in items if s == "CRIT")
    warns = sum(1 for s, *_ in items if s == "WARN")
    v, color, vt, action = verdict(crits, warns)
    rows = []
    for sev, sec, msg in sorted(items, key=lambda x: {"CRIT":0,"WARN":1,"INFO":2,"OK":3}.get(x[0],9)):
        if sev == "INFO" and (sec == "file" or "sha256=" in msg or "size=" in msg):
            continue  # fingerprint info goes in the header instead
        plain = plain_for(sev, sec, msg)
        explanation = f'<p class="plain">{html.escape(plain)}</p>' if plain and sev in ("CRIT","WARN") else ""
        rows.append(f"""
        <tr class="{sev.lower()}">
          <td><span class="badge" style="background:{SEV_COLOR[sev]}">{SEV_LABEL[sev]}</span></td>
          <td><span class="sec">{html.escape(sec)}</span>
              <div class="msg">{html.escape(msg)}</div>{explanation}</td>
        </tr>""")
    meta = {m.split("=",1)[0]: m.split("=",1)[1] for _, sec, m in items
            if sec == "file" and "=" in m}
    # model identity card: pull the meta KVs users can read at a glance
    id_rows = []
    for _, sec, m in items:
        if sec == "meta" and ("=" in m and ("architecture=" in m or m.split("=")[0] in
                ("general.quantized_by", "general.base_model", "general.source.url",
                 "model_type", "torch_dtype", "transformers_version", "base_model",
                 "quantized_by", "converted_by", "source"))):
            id_rows.append(m)
    arch_line = next((m for m in id_rows if "architecture=" in m), None)
    arch = arch_line.split("architecture=")[1].split()[0] if arch_line else "?"
    quant = next((m.split("=",1)[1] for s, sec, m in items
                  if sec == "meta" and m.startswith("quant=")), "")
    prov = [m for m in id_rows if "architecture=" not in m]
    tpl_findings = [(s, m) for s, sec, m in items if sec == "template"]
    tpl_crits = sum(1 for s, _ in tpl_findings if s == "CRIT")
    tpl_warns = sum(1 for s, _ in tpl_findings if s == "WARN")
    tpl_sources = [m.split("template sidecar: ", 1)[1].split(" (")[0] for s, m in tpl_findings
                   if s == "INFO" and m.startswith("template sidecar: ")]
    if tpl_crits:
        tpl_verdict, tpl_color = "❌ HOSTILE PATTERNS FOUND", SEV_COLOR["CRIT"]
    elif any("no chat template sidecar" in m.lower() for s, m in tpl_findings):
        tpl_verdict, tpl_color = "➖ NO SIDECAR TEMPLATE FOUND", SEV_COLOR["WARN"]
    elif any("no chat template" in m.lower() or "no tokenizer.chat_template" in m for s, m in tpl_findings):
        tpl_verdict, tpl_color = "➖ NONE BAKED INTO FILE", SEV_COLOR["WARN"]
    elif tpl_warns:
        tpl_verdict, tpl_color = "⚠️ SUSPICIOUS PATTERNS — READ BELOW", SEV_COLOR["WARN"]
    else:
        tpl_verdict, tpl_color = "✅ CLEAN", SEV_COLOR["OK"]
    idcard = f"""
      <table class="meta idcard">
        <tr><td>Model type</td><td>{html.escape(arch)}{(' · quant ' + html.escape(quant)) if quant else ''}</td></tr>
        {''.join(f'<tr><td>{html.escape(m.split("=")[0].replace("general.",""))}</td><td>{html.escape(m.split("=",1)[1])}</td></tr>' for m in prov)}
        <tr><td>Chat template</td><td style="color:{tpl_color};font-weight:600">{tpl_verdict}</td></tr>
        {f'<tr><td>Template source</td><td class="mono">{html.escape(", ".join(tpl_sources))}</td></tr>' if tpl_sources else ''}
      </table>"""
    return f"""
    <section class="fileblock">
      <h2>{html.escape(name)}</h2>
      <div class="verdict" style="border-color:{color}">
        <div class="vtitle" style="color:{color}">{v}</div>
        <div class="vtext">{vt}</div>
        <div class="vaction"><b>▶ {action}</b></div>
      </div>
      {idcard}
      <table class="meta">
        <tr><td>SHA-256 fingerprint</td><td class="mono">{html.escape(meta.get('sha256','?'))}</td></tr>
        <tr><td>Size</td><td>{html.escape(meta.get('size','?'))} bytes</td></tr>
      </table>
      <table class="findings">{''.join(rows)}</table>
    </section>"""

CSS = """
body { font-family: -apple-system, 'Segoe UI', sans-serif; max-width: 980px;
       margin: 0 auto; padding: 24px; color: #1a1a2e; background: #fafafa; }
h1 { border-bottom: 3px solid #2c3e50; padding-bottom: 8px; }
.fileblock { background: #fff; border: 1px solid #d5d8dc; border-radius: 10px;
             padding: 20px 24px; margin: 24px 0; }
.verdict { border-left: 6px solid; padding: 10px 16px; margin: 14px 0; background: #f8f9fa; border-radius: 4px; }
.vtitle { font-size: 1.35em; font-weight: 700; }
.vtext { color: #444; margin-top: 4px; }
.vaction { margin-top: 8px; padding: 8px 12px; background: #fff; border: 1px dashed #bbb;
           border-radius: 6px; font-size: 0.92em; }
.meta td { padding: 4px 12px 4px 0; font-size: 0.9em; color: #555; }
.mono { font-family: monospace; word-break: break-all; }
.findings { width: 100%; border-collapse: collapse; margin-top: 12px; }
.findings td { padding: 10px 8px; border-top: 1px solid #eee; vertical-align: top; }
.badge { color: #fff; font-size: 0.72em; font-weight: 700; padding: 3px 8px; border-radius: 10px; white-space: nowrap; }
.sec { font-weight: 600; font-size: 0.85em; text-transform: uppercase; letter-spacing: 0.4px; color: #666; }
.msg { font-family: monospace; font-size: 0.88em; margin-top: 3px; word-break: break-word; }
.plain { font-size: 0.92em; color: #333; margin: 6px 0 0; background: #f4f6f7; padding: 8px 10px; border-radius: 6px; }
tr.crit td:first-child { border-left: 4px solid #c0392b; }
tr.warn td:first-child { border-left: 4px solid #d68910; }
tr.ok td:first-child { border-left: 4px solid #1e8449; }
.foot { color: #777; font-size: 0.85em; margin-top: 30px; border-top: 1px solid #ddd; padding-top: 12px; }
.summary { display: flex; gap: 16px; margin: 20px 0; }
.sumitem { flex: 1; background: #fff; border: 1px solid; border-left-width: 6px; border-radius: 8px;
           padding: 12px 16px; text-align: center; }
.sumnum { display: block; font-size: 1.8em; font-weight: 800; }
.sumitem { color: #555; font-size: 0.85em; }
.idcard td { font-size: 0.95em; }
"""

VERSION = "0.3.0"

def generate(models, outdir):
    outdir.mkdir(parents=True, exist_ok=True)
    blocks, all_json = [], []
    tally = {"CRIT": 0, "WARN": 0, "OK": 0}
    for g in models:
        rep = Report()
        try:
            _audit(Path(g), rep)
        except Exception as e:
            rep.add("CRIT", "audit", f"audit failed: {e}")
        name = Path(g).name
        crits_n = sum(1 for s, *_ in rep.items if s == "CRIT")
        warns_n = sum(1 for s, *_ in rep.items if s == "WARN")
        tally["CRIT" if crits_n else ("WARN" if warns_n else "OK")] += 1
        blocks.append(render_file_block(name, rep.items))
        all_json.append({"file": str(g),
                         "findings": [{"severity": s, "section": sec, "message": m}
                                      for s, sec, m in rep.items]})
    stamp = __import__("datetime").datetime.now().strftime("%Y-%m-%d %H:%M")
    n = len(models)
    summary = f"""
    <div class="summary">
      <div class="sumitem" style="border-color:{SEV_COLOR['CRIT']}"><span class="sumnum" style="color:{SEV_COLOR['CRIT']}">{tally['CRIT']}</span>do not run</div>
      <div class="sumitem" style="border-color:{SEV_COLOR['WARN']}"><span class="sumnum" style="color:{SEV_COLOR['WARN']}">{tally['WARN']}</span>use with awareness</div>
      <div class="sumitem" style="border-color:{SEV_COLOR['OK']}"><span class="sumnum" style="color:{SEV_COLOR['OK']}">{tally['OK']}</span>no red flags</div>
    </div>"""
    doc = f"""<!doctype html><html><head><meta charset="utf-8">
<title>Model Audit Report</title><style>{CSS}</style></head><body>
<h1>🛡️ Model Audit Report</h1>
<p class="foot" style="border:none;margin-top:0">{n} file{'s' if n != 1 else ''} audited · generated {stamp} by model-audit v{VERSION}</p>
{summary if n > 1 else ''}
{''.join(blocks)}
<p class="foot"><b>What this report can and cannot tell you:</b> these checks read the file's
metadata, chat template, and structure. They prove what the file <i>claims</i> to be and whether
anything hostile hides in the instructions it carries. They cannot prove the mathematical weights
themselves are free of subtle tampering — for that, generate per-tensor fingerprints
(<code>--tensor-hashes</code>) and compare against a build you trust.</p>
</body></html>"""
    out = outdir / f"audit-report-{stamp.replace(' ', '_').replace(':', '')}.html"
    out.write_text(doc)
    (outdir / (out.stem + ".json")).write_text(json.dumps(all_json, indent=2))
    return out

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+")
    ap.add_argument("-o", "--outdir", type=Path, default=Path(__file__).parent / "reports")
    a = ap.parse_args()
    print(generate([Path(g) for g in a.models], a.outdir))
