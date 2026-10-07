# model-audit

Supply-chain auditor for (abliterated) model artifacts. Born from the observation that
"uncensored" builds from small uploaders are exactly where a hostile actor would hide
tampering — and the average user has no way to check.

Currently audits **GGUF** files. (Renamed from `gguf-audit`, which the GitHub redirect
still resolves; the scope never was GGUF-specific — see the roadmap below.)

## Layout
- `model_audit.py` — core scanner (CLI, scriptable, exit codes 0/1/2)
- `report.py` — human-readable HTML report generator (plain language, for non-technical users)
- `reports/` — generated audit reports (HTML + JSON)
- `templates/benign/` — honest reference templates
- `templates/tampered/` — attack specimens (documentation only — never install)
- `baselines/` — per-tensor fingerprint baselines for comparison
- `docs/` — write-ups

## Usage
```bash
# full audit of one file
python3 model_audit.py <file.gguf>

# also write per-tensor SHA-256 baseline (for comparing builds)
python3 model_audit.py <file.gguf> --tensor-hashes

# cross-check local hash against the Hugging Face repo it came from
python3 model_audit.py <file.gguf> --hf <uploader>/<repo>

# batch a whole collection
for f in /path/to/models/*/*.gguf; do python3 model_audit.py "$f"; done

# compare two tensor-fingerprint baselines (see docs/safe-workflow.md)
python3 model_audit.py --tensor-hashes model.gguf          # save baseline
python3 model_audit.py --diff model.old.json model.new.json

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

## Roadmap
- **safetensors support** — same checks on `.safetensors` (tensor offsets live in the
  header, so per-tensor hashing is cheaper), plus the sidecar `chat_template.jinja` /
  `tokenizer_config.json` template scan. Auditing upstream safetensors is also what lets
  you later prove a GGUF was derived from a given source tree.
- **Statistical per-tensor profiling** — compare dequantized block statistics against the
  profile a quant type should produce, as a hint at post-quantization training.
- **Baseline diff against a trusted self-quant** — localize edits to abliteration sites
  (`ffn_down`/`ffn_out`/`attn_o`) vs. spread changes (broad re-training).

---

**model-audit** — created by Exile, 2026. MIT licensed.
