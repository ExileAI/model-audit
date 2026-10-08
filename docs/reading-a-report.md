# How to read a model-audit report

You don't need to know anything about model files to understand a report. It
answers one question — *is there anything hidden in this file that could hurt me?*
— and it is careful about what it claims.

## The verdict, at the top of each file

Every audited file gets one of three banners:

| banner | meaning | what to do |
|---|---|---|
| **NO RED FLAGS FOUND** | nothing hostile was found in the metadata, chat template, or structure | nothing required; save a fingerprint baseline if the model matters to you |
| **USE WITH AWARENESS** | something needs your attention, but the file isn't clearly hostile | read each warning's plain-language note; many are false positives from tool-calling models |
| **DO NOT RUN THIS FILE** | a critical problem was found | delete or quarantine it; get the model from a source that documents its build, or make it yourself |

The three are a *scale of evidence*, not a scale of blame. "NO RED FLAGS" means
nothing hostile was found — it is **not** a promise that the weights are safe, and
**not** a statement that the uploader is trustworthy. See the bottom of the report.

## The identity card

A small table describes what the file *claims* to be:

- **Model type** — the architecture and quantization (e.g. `gemma4 · quant Q4_K_M`).
  A mismatch with the filename or the model card is worth noticing.
- **Chat template** — whether the instructions the model runs on are clean.
- **SHA-256 fingerprint** and **Size** — the file's identity. Write these down; if
  the file ever changes, they won't match.

## The finding levels

Below the banner, every finding is tagged:

- 🔴 **Critical** — a concrete hostile pattern (e.g. invisible characters hidden in
  the chat template). These set the verdict to DO NOT RUN.
- 🟠 **Warning** — something unusual that deserves a look. Often benign; read the
  plain-language note under it before worrying.
- ⚪ **Info** — a fact about the file, recorded for completeness. Not a problem.

Warnings and Info lines each carry a one-sentence explanation in plain language —
that note is usually all you need.

## What the report can and cannot tell you

- It **can** show what the file claims to be, whether anything hostile hides in the
  instructions it carries, and whether its structure contradicts itself.
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
