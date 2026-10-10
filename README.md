# model-audit

[![self-test](https://github.com/ExileAI/model-audit/actions/workflows/test.yml/badge.svg)](https://github.com/ExileAI/model-audit/actions/workflows/test.yml)
[![license: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/)

**Before you run a model someone else made, check what's inside it.**

`model-audit` is a supply-chain auditor for model files. You point it at a `.gguf` or
`.safetensors` file you downloaded, and reports known suspicious patterns, inconsistent metadata, and structural
problems in the inputs it successfully inspects. It then prints a plain-language
report; it cannot establish that a model is safe.

It exists because "uncensored" builds from small uploaders are exactly where a hostile
actor would hide tampering, and the average user has no way to check.

- Audits **GGUF** *and* **safetensors** — the format is auto-detected from the file's
  first bytes; no flag to remember.
- `.safetensors` auditing needs **no extra packages** (standard-library reader only).
- The report is written for non-technical readers, not for security engineers.
- Local checks read your files without uploading them. The optional `--hf` check
  queries Hugging Face metadata for the repository you specify.

> **Not comfortable with a terminal?** You need three commands, they are copy-paste
> below, and the report opens in your web browser like any other web page. That's the
> whole workflow.

*(Renamed from `gguf-audit`; the GitHub redirect still resolves.)*

## Install (one time)

```bash
git clone https://github.com/ExileAI/model-audit.git
cd model-audit
pip install -r requirements.txt
```

That's the whole install. The full install needs `python3` 3.10 or newer because
`gguf` 0.19.0 requires Python 3.10+; CI runs 3.11. The
`pip install` pulls in the GGUF library and the Hugging Face client used by `--hf` — if
you only audit `.safetensors` files you can skip it entirely.

## Quick start

**1 — Audit one file** (works out of the box, prints findings to the terminal):

```bash
python3 model_audit.py ~/Downloads/my-model.gguf
```

**2 — Or get a readable report** (writes an HTML file you open in your browser):

```bash
python3 report.py ~/Downloads/my-model.gguf
```

The HTML/JSON pair lands in a unique run directory under `reports/`; the CLI prints
the HTML path. Finding exit codes 1 and 2 can accompany a successfully written report. Open it the way you'd open any web page —
double-click it, or drag it into your browser. Everything below is an optional extra.

### Worked examples — both formats

Copy-paste, with made-up names. **GGUF** (a single file):

```bash
# audit a quantized GGUF
python3 model_audit.py ~/Downloads/Example-7B-Uncensored-Q4_K_M.gguf

# same file, as a readable report saved to reports/
python3 report.py ~/Downloads/Example-7B-Uncensored-Q4_K_M.gguf
```

**safetensors** (a weights file plus its sidecars in the same folder):

```bash
# keep the sidecars beside the weights — the chat template lives in
# chat_template.jinja / tokenizer_config.json, and the audit reads them there
python3 model_audit.py ~/Downloads/Example-7B-Uncensored/model.safetensors

python3 report.py ~/Downloads/Example-7B-Uncensored/model.safetensors
```

> **Using LM Studio?** Your models already live under `~/.lmstudio/models/<uploader>/<repo>/`.
> Point the commands straight at the file there, or `cd` into that folder first.

### What the verdict means

Every file gets one of three banners, in plain language:

| banner | in one line |
|---|---|
| ✅ **NO RED FLAGS IN COMPLETED CHECKS** | no known suspicious patterns in the inputs successfully inspected |
| ⚠️ **WARNINGS: REVIEW THE DETAILS** | read the specific warnings and their explanations |
| 🛑 **CRITICAL FINDINGS: REVIEW BEFORE LOADING** | a critical finding needs investigation before loading |

"No red flags" describes only completed checks — it is **not** a promise the weights are
safe, and **not** a statement about the uploader. See `docs/reading-a-report.md` for a
full plain-language walkthrough, and `examples/` for legacy sample reports.

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
- `examples/` — committed **legacy sample** reports with older banner wording,
  plus the scripts that regenerate them (`examples/README.md`)

## Full usage

```bash
# full audit of one file (GGUF or safetensors)
python3 model_audit.py <file.gguf>
python3 model_audit.py <model.safetensors>

# write a baseline without replacing an existing reference
python3 model_audit.py <file> --tensor-hashes
# after a re-download, preserve the original and choose a new destination
python3 model_audit.py <file> --tensor-hashes --baseline-out model.new.json

# cross-check local hash against the Hugging Face repo it came from
python3 model_audit.py <file> --hf <uploader>/<repo> --revision <commit_sha>

# batch a whole collection
for f in /path/to/models/*/*.gguf; do python3 model_audit.py "$f"; done

# compare two tensor-fingerprint baselines (see docs/safe-workflow.md)
python3 model_audit.py --diff model.old.json model.new.json

# compare fingerprint evidence across formats (not proof of conversion history)
python3 model_audit.py <model.safetensors> --tensor-hashes    # the source
python3 model_audit.py <model-Q8_0.gguf> --tensor-hashes      # the derived build
python3 model_audit.py --diff <model.safetensors>.tensorhashes.json \
                             <model-Q8_0.gguf>.tensorhashes.json

# human-readable HTML report (one per file, plain language)
python3 report.py <file1> <file2>

# print the tool version
python3 model_audit.py --version
```

## Exit codes
| code | meaning |
|---|---|
| `0` | clean — no CRIT and no WARN |
| `1` | findings — at least one WARN (including an existing baseline destination), no CRIT |
| `2` | a CRIT finding or hard error, such as a file that could not be read |

Both scripts expose `--version`. Scripts and CI can branch on the exit code; the
per-tensor diff mode (`--diff`) uses the same scale.

## Evidence rubric

HTML and JSON include four dimensions: **Structure**, **Template inspection**,
**Provenance evidence**, and **Baseline comparison**. Each records PASS, CONCERN,
FAIL, NOT CHECKED, or justified N/A, with a reason and scope. Requested-check
coverage is shown separately as COMPLETE or INCOMPLETE. There is no numeric safety
score: four passes would still not prove safe weights.

- PASS is scoped agreement or a completed inspection, never automatic uploader trust
- CONCERN means suspicious or inconclusive evidence, not a proven attack
- FAIL means an established invalid structure or unambiguous reference mismatch
- NOT CHECKED distinguishes absent, unrequested, failed, and incomplete inspection
- N/A requires a specific exemption, such as an embedded template in a clip/mmproj file

Finding severity remains independent: a critical signature can show CONCERN in the
rubric while retaining exit 2. Optional unrequested checks do not change exit status.
Same-format baseline agreement covers bytes and recorded shapes only; dtype, byte
order, and tensor roles remain NOT VERIFIED. Cross-format PASS requires nonempty,
full one-to-one correspondence by bytes or supported typed normalized fingerprints;
it does not check shapes, tensor roles, numerical/model equivalence, or history.
Unknown formats, ambiguous candidates, empty inventories, and partial matches remain
CONCERN. A CONCERN can accompany INFO findings and exit 0; read the rubric as well
as the exit code. Generating a baseline is not comparing one. See
[the report guide](docs/reading-a-report.md).

The core `--json` output remains one findings array. Reports retain `file` and
`findings`, with additive rubric data. Existing baseline files remain readable;
baseline writes refuse existing destinations with WARN and exit 1 (exit 2 if
another finding is CRIT). Use `--baseline-out` for a new snapshot and retain the
original reference.

## Sample reports
`examples/` holds committed legacy reports generated from real files. Their older
banners and explanations are historical samples, not the current wording or
evidence contract described above (click to open):

- [`report-01-clean-gemma-12b-obliterated`](examples/report-01-clean-gemma-12b-obliterated.html) — **NO RED FLAGS**
- [`report-02-warn-mxfp4-anonymous`](examples/report-02-warn-mxfp4-anonymous.html) — **USE WITH AWARENESS**
- [`report-03-aeon-pair-two-uploaders`](examples/report-03-aeon-pair-two-uploaders.html) — two uploaders, one "AEON" lineage
- [`report-04-do-not-run-template-swap`](examples/report-04-do-not-run-template-swap.html) — **DO NOT RUN**, produced by swapping the
  chat template for a hostile one. The weights were unchanged in this example;
  the sample shows what a single replaced string does to the verdict, without
  establishing that the weights are safe.

See `examples/README.md` for a plain-language walkthrough of each.

## What each format is checked for

| | GGUF | safetensors |
|---|---|---|
| Identity | file SHA-256, size | file SHA-256, size |
| Metadata | `general.*` KVs, quant vs filename, tokenizer KVs | header layout, `__metadata__`, sidecar `config.json` provenance |
| Structure | tensor census, zero-dim blocks | header/tensor self-consistency (dtype × shape vs declared bytes), overlap, gaps, trailing bytes, shard `*.index.json` consistency |
| Chat template | `tokenizer.chat_template` inside the file | **sidecars only**: `chat_template.jinja`, `chat_template` in `tokenizer_config.json` (all entries), conflict between the two |
| Code execution path | — | `auto_map` in config/tokenizer_config, shipped scripts |
| Per-tensor hashes | `--tensor-hashes` | `--tensor-hashes` (cheaper — offsets come from the header; both record value-level hashes for supported tensors, see below) |
| Cross-format comparison | `--diff`: byte matches, then typed normalized matches, then inconclusive legacy/ambiguous candidates; each tensor used once | Quantized nonmatches are INFO and remain inconclusive, not evidence of tampering |
| Remote check | `--hf`, by LFS SHA-256 first | same, any LFS-tracked file |
| Uploader checksums | a shipped `MANIFEST.txt`/`SHA256SUMS`/`*.sha256` next to the file: does it still agree with this file, or has the file been renamed under it | nothing about safety — it is the uploader's own claim. Agreement is internal consistency; disagreement is the drift a swap leaves behind |

Safetensors empty tensors are structurally legal and baseline-eligible. They receive
an inventory-heuristic WARN for review, not a structural failure.

### Per-tensor baseline evidence

The existing `tensors` (byte hashes), `values` (normalized hashes), and `shapes` maps
are preserved. New baselines add optional per-tensor `value_metadata` with `dtype`,
`byte_order: "little"`, and `normalization: "f32-le-bits-v1"`. Supported little-endian
BF16 and F32 tensors get explicit `values` entries, including F32 itself.
Big- or unknown-endian GGUF tensors retain byte hashes only, with no value claims.
F16's existing struct-based normalized hash is retained only as a legacy/inconclusive
candidate because that conversion does not preserve NaN payload bits. Validation
rejects typed F32 metadata whose byte and normalized hashes contradict each other.

An **exact normalized-F32-fingerprint match** requires value-to-value digests and
supported metadata on both sides. BF16 widening to F32 is mathematically exact;
matching fingerprints establish consistency of the canonical normalized bitstream,
not numerical/model equivalence or a lossless whole-model conversion history.
Container labels alone cannot identify the dtype of a raw hash. For example, an
older BF16 value hash may equal the raw hash of I32 bits. Such legacy or ambiguous
matches stay **INCONCLUSIVE**, even when the file formats are known.

## Threat model & limits
- Metadata/template/structure checks look for known suspicious signatures, label
  mismatches, missing provenance, sidecar executables, reference mismatches, and
  contradictory file structures. Incomplete checks are not a clean result.
- A safetensors header is text the uploader wrote. It agreeing with the file's length and
  with itself proves consistency, not safety — the weights themselves are still unaudited.
- Per-tensor hashes catch: swapped or edited weight blocks **when compared against a
  trusted baseline of the same base model**.
- Cross-format comparison matches byte hashes first, then supported typed normalized
  fingerprints, then ambiguous candidates. Each tensor is consumed at most once and
  both unmatched sides are reported (complete counts and up to eight names each).
  Only supported value-to-value matches carry
  the normalized-F32-fingerprint claim; ambiguous candidates are WARN/INCONCLUSIVE.
  General comparison limits and quantized nonmatches are INFO. Neither a match nor
  a scoped PASS verifies shapes, tensor roles, model equivalence, or conversion history.
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
  so recompute them and compare against the scales stored in the file. Until then, matching
  unchanged tensors provide limited fingerprint evidence, not proof of a source lineage.
- **Remote-only audit** — audit a Hugging Face repo's file list and hashes straight from the
  API, without downloading multi-GB files.

## Changelog
Newest first. Each entry is a commit on `master` (`git show <sha>` for the diff).

| commit | change |
|---|---|
| `cc17289` | **v0.4.0** — sample reports committed in `examples/`; `--version` on both scripts; `docs/reading-a-report.md` |
| `7184447` | gitignore: ignore `*.safetensors` and `*.safetensors.index.json` |
| `81e4569` | report: show the audited format, fix dead provenance rows, drop internal jargon |
| `e501a5b` | docs: layout, conversion-comparison usage, and the workflow step |
| `da55e18` | docs: value-level comparison and what a quantized build can still be tied to |
| `0aebb59` | introduced value-level fingerprints (historical lossless-conversion claim is limited by the evidence contract above) |
| `cc929ee` | cross-format content match: key on bytes, report shape separately |
| `7677f4d` | verify uploader-supplied checksum files |
| `d7d3a58` | cross-format diff: compare by content, not by name |
| `59d1260` | add safetensors support |
| `6750cc6` | attribution: Exile → ExileAI |
| `95d4631` | rename `gguf-audit` → `model-audit` |
| `bc8c3d7` | CI: run scanner self-tests on every push/PR |
| `5d2e45a` | initial release: GGUF supply-chain auditor |

---

**model-audit** — created by ExileAI, 2026. MIT licensed.
