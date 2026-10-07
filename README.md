# gguf-audit

Supply-chain auditor for (abliterated) GGUF models. Born from the observation that
"uncensored" GGUFs from small uploaders are exactly where a hostile actor would hide
tampering — and the average user has no way to check.

## Layout
- `gguf_audit.py` — core scanner (CLI, scriptable, exit codes 0/1/2)
- `report.py` — human-readable HTML report generator (plain language, for non-technical users)
- `reports/` — generated audit reports (HTML + JSON)
- `templates/benign/` — honest reference templates
- `templates/tampered/` — attack specimens (documentation only — never install)
- `baselines/` — per-tensor fingerprint baselines for comparison
- `docs/` — write-ups

## Usage
```bash
# full audit of one file
python3 gguf_audit.py <file.gguf>

# also write per-tensor SHA-256 baseline (for comparing builds)
python3 gguf_audit.py <file.gguf> --tensor-hashes

# cross-check local hash against the Hugging Face repo it came from
python3 gguf_audit.py <file.gguf> --hf <uploader>/<repo>

# batch a whole collection
for f in /path/to/models/*/*.gguf; do python3 gguf_audit.py "$f"; done

# compare two tensor-fingerprint baselines (see docs/safe-workflow.md)
python3 gguf_audit.py --tensor-hashes model.gguf          # save baseline
python3 gguf_audit.py --diff model.old.json model.new.json

# human-readable HTML report (one per file, plain language)
python3 report.py <file1.gguf> <file2.gguf>
```

## Threat model & limits
- Metadata/template/structure checks catch: hostile chat templates, hidden instructions,
  label mismatches, missing provenance, sidecar executables, and remote file swaps.
- Per-tensor hashes catch: swapped or edited weight blocks **when compared against a
  trusted baseline of the same base model**.
- NOTHING here can prove weights are benign in isolation. Hashes prove consistency,
  not safety. The gold standard remains: quantize yourself from a source you trust.

---

**gguf-audit** — created by Exile, 2026. MIT licensed.
