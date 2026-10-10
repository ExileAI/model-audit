# How to read a model-audit report

You don't need to know anything about model files to understand a report. It
shows the evidence the scanner collected and which checks completed. It cannot
answer whether a model is safe in general.

## The verdict, at the top of each file

Every audited file gets one of three banners:

| banner | meaning | what to do |
|---|---|---|
| **NO RED FLAGS IN COMPLETED CHECKS** | completed checks found no known suspicious patterns; read coverage too | review any uninspected areas; save a fingerprint baseline if comparison matters |
| **WARNINGS: REVIEW THE DETAILS** | something needs your attention | read each warning's plain-language explanation |
| **CRITICAL FINDINGS: REVIEW BEFORE LOADING** | a critical finding needs investigation | resolve it before loading; it does not establish malicious intent |

The three summarize finding severity, not blame. "NO RED FLAGS IN COMPLETED CHECKS"
means no known suspicious patterns were found in completed checks — it is **not**
a promise that the weights are safe, and **not** a statement that the uploader is
trustworthy. See the bottom of the report.

## Evidence rubric and coverage

Four rows separate Structure, Template inspection, Provenance evidence, and
Baseline comparison. PASS means a specific check completed and matched its
criteria; CONCERN means suspicious or inconclusive evidence; FAIL means a
confirmed structural/reference contradiction. NOT CHECKED records why a check
was absent, unrequested, unavailable, or incomplete. N/A is an explicit exemption.

COMPLETE/INCOMPLETE describes requested applicable coverage, not model safety.
Optional unrequested remote checks do not count as failures. A template signature
can be CONCERN while its original severity remains Critical. A checksum PASS
means only agreement with the named source, not that the source is trusted.
Same-format baseline PASS is limited to recorded bytes and shapes: dtype, byte
order, tensor roles, and model behavior remain unverified. Cross-format PASS has
a different scope: nonempty, full one-to-one correspondence by bytes or supported
typed normalized fingerprints. It does not check shapes, semantic roles,
numerical/model equivalence, or conversion history. Unknown formats, ambiguous
candidates, empty inventories, and partial matches remain CONCERN. There is no
overall numeric score.

Cross-format comparisons use byte matches first, typed normalized matches second,
and ambiguous candidates last. Each tensor is used at most once; unmatched tensors
on both sides are reported with complete counts and up to eight names each.
An **exact normalized-F32-fingerprint match** requires
explicit normalized hashes and supported dtype, little-endian byte order, and
`f32-le-bits-v1` metadata on both sides, for BF16 or F32. BF16 widening to F32 is
mathematically exact; a matching fingerprint establishes consistency of the
normalized bitstream, not a lossless whole-model conversion or historical lineage.
F16's older normalization remains an inconclusive candidate because it does not
preserve NaN payload bits. Contradictory byte and normalized hashes for typed F32
metadata are rejected when validating a baseline.

Older baselines remain readable, but an untyped normalized hash matching a raw
hash is **INCONCLUSIVE**. A known file format does not tell us the raw tensor's
dtype; I32 bits, for example, could match a BF16 normalized fingerprint. Actual
ambiguous candidates and empty inventories get Warning findings. General comparison
limits and quantized nonmatches are Info; a partial quantized comparison can be
CONCERN even with exit 0 and does not itself suggest tampering. Big- or unknown-endian
GGUF tensors provide byte fingerprints only, without normalized-value claims.

## The identity card

A small table describes what the file *claims* to be:

- **Model type** — the architecture and quantization (e.g. `gemma4 · quant Q4_K_M`).
  A mismatch with the filename or the model card is worth noticing.
- **Chat template** — whether supported instructions were inspected and known
  suspicious patterns were found; uninspected is never clean.
- **SHA-256 fingerprint** and **Size** — the file's identity. Write these down; if
  the file ever changes, they won't match.

## The finding levels

Below the banner, every finding is tagged:

- 🔴 **Critical** — a critical signal, invalid structure, or hard error (for example,
  invisible template characters). These set CRITICAL FINDINGS: REVIEW BEFORE LOADING;
  they do not prove intent.
- 🟠 **Warning** — something unusual that deserves a look. Often benign; read the
  plain-language note under it before worrying.
- ⚪ **Info** — context or a limit on the comparison, rather than a warning by itself.

Warnings and Info lines each carry a one-sentence explanation in plain language —
that note is usually all you need.

An empty safetensors tensor is structurally legal and can be fingerprinted for a
baseline, but receives an inventory-heuristic Warning worth reviewing. An existing
baseline output destination also gets a Warning and exit 1, unless an unrelated
Critical finding sets exit 2. It is never overwritten; choose a fresh destination
with `--baseline-out`.

## What the report can and cannot tell you

- It **can** show self-reported labels, known suspicious instruction patterns,
  structural contradictions, and the scope actually inspected.
- It **cannot** prove the mathematical weights are free of subtle tampering. For
  that, generate per-tensor fingerprints (`--tensor-hashes`) and compare against a
  build you trust.
- It **never** tells you who deserves credit for a model. It reports facts; the
  judgment is yours.

## A worked example

The committed files under `examples/` are legacy samples with older banners and
explanations. `examples/report-04-do-not-run-template-swap.html` has the historical
DO NOT RUN banner: its chat template was replaced with a hostile one, changing
nothing else. The example does not establish the safety of the unchanged weights.
Compare it with
`examples/report-01-clean-gemma-12b-obliterated.html`: same model, opposite
verdicts. It is the clearest possible illustration of what the audit is actually
looking at — and of how much one metadata string can matter.
