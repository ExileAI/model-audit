# How to read a model-audit report

You don't need to know anything about model files to understand a report. It
shows the evidence the scanner collected and which checks completed. It cannot
answer whether a model is safe in general.

## The verdict, at the top of each file

Every audited file gets one of three banners:

| banner | meaning | what to do |
|---|---|---|
| **NO RED FLAGS FOUND** | completed checks found no known suspicious patterns; read coverage too | nothing required; save a fingerprint baseline if the model matters to you |
| **USE WITH AWARENESS** | something needs your attention, but the file isn't clearly hostile | read each warning's plain-language note; many are false positives from tool-calling models |
| **DO NOT RUN THIS FILE** | a critical problem was found | delete or quarantine it; get the model from a source that documents its build, or make it yourself |

The three are a *scale of evidence*, not a scale of blame. "NO RED FLAGS" means
nothing hostile was found — it is **not** a promise that the weights are safe, and
**not** a statement that the uploader is trustworthy. See the bottom of the report.

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
Baseline PASS is limited to recorded bytes and shapes: dtype, byte order, tensor
roles, and model behavior remain unverified. There is no overall numeric score.

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
  invisible template characters). These set DO NOT RUN; they do not prove intent.
- 🟠 **Warning** — something unusual that deserves a look. Often benign; read the
  plain-language note under it before worrying.
- ⚪ **Info** — a fact about the file, recorded for completeness. Not a problem.

Warnings and Info lines each carry a one-sentence explanation in plain language —
that note is usually all you need.

## What the report can and cannot tell you

- It **can** show self-reported labels, known suspicious instruction patterns,
  structural contradictions, and the scope actually inspected.
- It **cannot** prove the mathematical weights are free of subtle tampering. For
  that, generate per-tensor fingerprints (`--tensor-hashes`) and compare against a
  build you trust.
- It **never** tells you who deserves credit for a model. It reports facts; the
  judgment is yours.

## A worked example

`examples/report-04-do-not-run-template-swap.html` is a DO NOT RUN report — on a
model that is **not** suspect. We took a clean model and replaced only its chat
template with a hostile one, changing nothing else. Compare it with
`examples/report-01-clean-gemma-12b-obliterated.html`: same model, opposite
verdicts. It is the clearest possible illustration of what the audit is actually
looking at — and of how much one metadata string can matter.
