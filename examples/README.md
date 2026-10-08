# Sample reports

Real audits of real files, kept here so you can see exactly what the tool prints
before you run it on anything of your own. Open the `.html` files in a browser;
the matching `.json` is the machine-readable form of the same report.

| # | sample | what it shows |
|---|---|---|
| 01 | `report-01-clean-gemma-12b-obliterated` | a well-attributed model that audits **NO RED FLAGS** |
| 02 | `report-02-warn-mxfp4-anonymous` | a **USE WITH AWARENESS** result — warnings, no criticals (uploader not named) |
| 03 | `report-03-aeon-pair-two-uploaders` | two uploaders publishing the same "AEON" lineage, both clean |
| 04 | `report-04-do-not-run-template-swap` | a **DO NOT RUN** result — on a model that is **not** actually suspect. Read the note below. |

All four regenerate with `./regenerate_samples.sh`. Paths in the reports are
relative, so no personal directory ever appears. The uploader behind sample 02 is
deliberately not named — regenerate it yourself by pointing `WARN_SRC` at any file
that produces a warning.

---

## Sample 04 — what a harmful chat template looks like, and why it matters

**The model in sample 04 is not suspect, and nothing is wrong with its weights.**
It is the same clean Gemma-4-12B-OBLITERATED as sample 01, essentially byte for
byte, with one thing changed on purpose: we overwrote the **chat template** stored
inside the GGUF with one of the hostile templates from `templates/tampered/`, then
re-ran the audit.

Put the two reports side by side:

- sample **01** — the untouched model — **NO RED FLAGS FOUND**
- sample **04** — same weights, hostile template — **DO NOT RUN THIS FILE**

The weights did not change. The tensors, the metadata, the tensor count are all
the same. The entire difference is one metadata string — and it is enough to flip
the verdict from "run it" to "do not run it."

That is the whole point of the sample. A chat template is not inert documentation:
it is instructions that get assembled into **every prompt** the moment you chat
with the model — silently, before your first token. Hidden instructions, secret
exfiltration, sandbox escapes: a hostile template is exactly where those live. It
is why `model-audit` reads the template straight out of the file instead of
trusting the model card, and why a template swap alone is the difference between a
model you can run and one you should not.

So read sample 04 as a **reference image**, not an accusation: *this is what a
dangerous template looks like on paper.* The underlying model is fine.

*(Reproduce it: `synthesize_do_not_run.py` rewrites only the `tokenizer.chat_template`
metadata value and copies the tensor data through unchanged — the script's docstring
explains the GGUF layout details.)*

---

## Sample 03 — two uploaders, one "AEON" lineage

Plain language, because this one is easy to misread.

The two files come from different uploaders (`vcruz305`, `Abiray`) and both carry
the "AEON" name. Auditing them reports **facts only**:

- Both audit **NO RED FLAGS** — nothing hostile was found in either file.
- Their internal names differ: one records itself as
  `Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED`, the other as just `Raw_Model` — so one
  file still says what it is, while the other carries no record of its own origin
  inside the file.
- Neither file's audit surfaces a source / base-model field — the artifacts don't
  state where they came from.

What that does **not** mean:

- It is not proof that either copied the other.
- It is not a clean bill of health for originality.

"No red flags" means nothing hostile was found — not that the work is original.
An absent provenance string means the file doesn't say — not that the uploader is
hiding something. `model-audit` reports what the bytes say and stops there; deciding
who deserves credit is a human call, made from a citation trail, never from a
verdict banner. Following that trail automatically (file hashes, sidecars, commit
history, license compliance) is a separate, planned direction — see the roadmap in
the main README.
