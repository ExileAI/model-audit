# model-audit

Supply-chain auditor for (abliterated) model artifacts. Born from the observation that
"uncensored" builds from small uploaders are exactly where a hostile actor would hide
tampering — and the average user has no way to check.

Audits **GGUF** and **safetensors** files. The format is auto-detected from the file's
first bytes; no flag, no library needed for safetensors (stdlib only). (Renamed from
`gguf-audit`, which the GitHub redirect still resolves.)

## Layout
- `model_audit.py` — core scanner (CLI, scriptable, exit codes 0/1/2)
- `report.py` — human-readable HTML report generator (plain language, for non-technical users)
- `templates/benign/` — honest reference templates
- `templates/tampered/` — attack specimens (documentation only — never install)
- `tests/test_scanner.py` — chat-template scanner specimens
- `tests/test_safetensors.py` — structural specimens, synthesized byte by byte
- `tests/test_report.py` — report rendering: no dropped findings, no internal jargon
- `docs/limitations.md` — what the audit can and cannot prove (read before changing claims)
- `docs/safe-workflow.md` — the download → pin → audit → baseline recipe
- `docs/reading-a-report.md` — plain-language guide to the report, for non-technical readers
- `reports/` — your generated audit reports (HTML + JSON); gitignored
- `baselines/` — per-tensor fingerprint baselines for comparison; gitignored
- `examples/` — committed **sample** reports so you can see what to expect before
  you run anything, plus the scripts that regenerate them (`examples/README.md`)

## Usage
```bash
# full audit of one file (GGUF or safetensors)
python3 model_audit.py <file.gguf>
python3 model_audit.py <model.safetensors>

# also write per-tensor SHA-256 baseline (for comparing builds)
python3 model_audit.py <file> --tensor-hashes

# cross-check local hash against the Hugging Face repo it came from
python3 model_audit.py <file> --hf <uploader>/<repo> --revision <commit_sha>

# batch a whole collection
for f in /path/to/models/*/*.gguf; do python3 model_audit.py "$f"; done

# compare two tensor-fingerprint baselines (see docs/safe-workflow.md)
python3 model_audit.py --diff model.old.json model.new.json

# did this build come from that source tree? audit both, then compare across formats
python3 model_audit.py <model.safetensors> --tensor-hashes    # the source
python3 model_audit.py <model-Q8_0.gguf> --tensor-hashes      # the derived build
python3 model_audit.py --diff <model.safetensors>.tensorhashes.json \
                             <model-Q8_0.gguf>.tensorhashes.json

# human-readable HTML report (one per file, plain language)
python3 report.py <file1> <file2>
```

## Exit codes
| code | meaning |
|---|---|
| `0` | clean — no CRIT and no WARN |
| `1` | findings — at least one WARN (or a `--diff` reported changes), no CRIT |
| `2` | hard error / **DO NOT RUN** — a CRIT finding, or a file that could not be read |

Both scripts expose `--version`. Scripts and CI can branch on the exit code; the
per-tensor diff mode (`--diff`) uses the same scale.

## Sample reports
`examples/` holds committed reports generated from real files, so you can see the
output before running anything:

- `report-01-clean-gemma-12b-obliterated` — **NO RED FLAGS**
- `report-02-warn-mxfp4-anonymous` — **USE WITH AWARENESS**
- `report-03-aeon-pair-two-uploaders` — two uploaders, one "AEON" lineage
- `report-04-do-not-run-template-swap` — **DO NOT RUN**, produced by swapping the
  chat template of an otherwise clean model for a hostile one. The weights are
  fine; the sample shows what a single replaced string does to the verdict.

See `examples/README.md` for a plain-language walkthrough of each.

## What each format is checked for

| | GGUF | safetensors |
|---|---|---|
| Identity | file SHA-256, size | file SHA-256, size |
| Metadata | `general.*` KVs, quant vs filename, tokenizer KVs | header layout, `__metadata__`, sidecar `config.json` provenance |
| Structure | tensor census, zero-dim blocks | header/tensor self-consistency (dtype × shape vs declared bytes), overlap, gaps, trailing bytes, shard `*.index.json` consistency |
| Chat template | `tokenizer.chat_template` inside the file | **sidecars only**: `chat_template.jinja`, `chat_template` in `tokenizer_config.json` (all entries), conflict between the two |
| Code execution path | — | `auto_map` in config/tokenizer_config, shipped scripts |
| Per-tensor hashes | `--tensor-hashes` | `--tensor-hashes` (cheaper — offsets come from the header; both record a value-level hash too, see below) |
| Cross-format comparison | `--diff`: byte-identical blobs first, then value-identical ones (a converter upcasting bf16 norms to f32 keeps the numbers, not the bytes) | Quantized blocks are not comparable this way — the diff says so instead of guessing |
| Remote check | `--hf`, by LFS SHA-256 first | same, any LFS-tracked file |
| Uploader checksums | a shipped `MANIFEST.txt`/`SHA256SUMS`/`*.sha256` next to the file: does it still agree with this file, or has the file been renamed under it | nothing about safety — it is the uploader's own claim. Agreement is internal consistency; disagreement is the drift a swap leaves behind |

## Threat model & limits
- Metadata/template/structure checks catch: hostile chat templates, hidden instructions,
  label mismatches, missing provenance, sidecar executables, remote file swaps, and
  self-contradicting or appended-to file structures.
- A safetensors header is text the uploader wrote. It agreeing with the file's length and
  with itself proves consistency, not safety — the weights themselves are still unaudited.
- Per-tensor hashes catch: swapped or edited weight blocks **when compared against a
  trusted baseline of the same base model**.
- A baseline also answers one provenance question directly: whether a conversion was
  lossless. A source → derived pair matches on byte-identical blobs *and* on value-identical
  blobs (a converter upcasting bf16 norms to f32 preserves the numbers, not the bytes), so
  `--diff` can show a build carries the same weights as a tree you audited — and it says
  outright when a comparison is not meaningful (block-quantized weights).
- A shipped checksum list (`MANIFEST.txt`, `SHA256SUMS`) is checked against the file, in both
  directions: agreement is internal consistency, a mismatch under the file's own name is a
  red flag, and the hash appearing under a *different* name means the file was renamed
  without its contents changing.
- NOTHING here can prove weights are benign in isolation. Hashes prove consistency,
  not safety. The gold standard remains: quantize yourself from a source you trust.

## Roadmap
- **Statistical per-tensor profiling** — compare dequantized block statistics against the
  profile a quant type should produce, as a hint at post-quantization training.
- **Baseline diff against a trusted self-quant** — localize edits to abliteration sites
  (`ffn_down`/`ffn_out`/`attn_o`) vs. spread changes (broad re-training).
- **Quantized lineage** — proving a Q4_K_M came from a specific fp source means re-deriving
  the quantization: K-quant block scales are a deterministic function of the source weights,
  so recompute them and compare against the scales stored in the file. Until then, a
  quantized build can only be tied to its source through the tensors quantization leaves
  untouched (norms), which the diff already reports.

---

**model-audit** — created by ExileAI, 2026. MIT licensed.