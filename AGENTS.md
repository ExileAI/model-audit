# AGENTS.md — model-audit

Guidance for AI coding agents working in this repository.

## What this project is

A supply-chain auditor for model artifacts — GGUF and safetensors. It reads the
artifact — metadata, chat template, tensor structure, hashes — and reports what it
can prove and what it cannot. The tool is built for non-technical users first
(plain-language HTML reports), with scriptable CLI flags for power users.

Core philosophy, reflected everywhere in the code: **hashes prove consistency, not
safety.** Never let a report, doc, or message imply a clean audit means the weights
are benign.

## Layout

```
model_audit.py       core scanner CLI (exit codes: 0 clean, 1 warnings, 2 critical)
report.py            HTML report generator (plain language, for non-technical users)
tests/test_scanner.py      self-test; tampered specimens must be caught, benign must pass
tests/test_safetensors.py  self-test; synthesized safetensors specimens, one broken invariant each
templates/tampered/  attack specimens — documentation only, never install
templates/benign/    honest reference templates
docs/limitations.md  what the audit can and cannot prove (read before changing claims)
reports/             generated output; gitignored, never commit
```

## Non-negotiable rules

1. **Honesty over reassurance.** Every finding, verdict, and README claim must respect
   `docs/limitations.md`. A tool that says "safe" gets people hurt; this tool says
   "no red flags" and explains the difference.
2. **Every finding needs a plain-language explanation.** If you add a scanner rule,
   add its non-technical translation to `PLAIN` in report.py. Unexplained warnings
   are how users learn to ignore the tool.
3. **False positives are as serious as false negatives.** A scanner that cries wolf
   gets ignored. Any pattern-tightening must pass the benign/tool-calling tests.
4. **Never weaken a test to make code pass.** The specimen tests encode real attack
   signatures. If a legit template trips them, fix the pattern, not the test.
5. **mmproj/clip files have no chat template by design.** Don't "fix" that INFO
   finding back into a CRIT.

## Scanner design notes

- Chat-template scan runs on comment-stripped text (`{#...#}` demoted) to avoid
  self-documenting comments tripping patterns. Match contexts are sliced from that
  same stripped text — slicing the raw template with an offset from the stripped one
  shows comment text in the excerpt and makes a real hit look like a false positive.
- "wording"-class findings (file I/O, exec/OS) are demoted to INFO when near
  tool-calling vocabulary — tool-calling templates legitimately discuss these.
- Remote check (`--hf`) matches by SHA-256 first, filename second, across any
  LFS-tracked file — upstream renames are common; the hash is the identity.
- `--revision` pins the HF commit; always recommend it, since force-pushes erase
  history.
- **safetensors carries no chat template.** Never copy the GGUF rule (missing template
  = CRIT) into that path: there, a missing sidecar template is INFO, because the format
  cannot carry one. What matters is scanning *every* template sidecar in the directory
  (`*.jinja`, and every entry of `chat_template` in `tokenizer_config.json`) and
  flagging when two of them disagree.
- safetensors structural checks are self-consistency checks: dtype × shape must equal
  the declared byte span, spans must not overlap, data must not run past EOF, and no
  bytes may be unaccounted for. Cheap, and they catch hand-edited files.
- Format dispatch is by magic bytes (`GGUF`, else 8-byte length + `{` JSON), never by
  filename extension. An absurd header length must stay a *finding*, not a reason to
  call the file unrecognized.

## Testing

```bash
python3 tests/test_scanner.py        # GGUF chat-template scanner specimens
python3 tests/test_safetensors.py    # safetensors structural specimens
python3 model_audit.py --help && python3 report.py --help
```

Both suites must be all green before any commit (CI runs them on push/PR).

New scanner patterns: add a specimen under templates/tampered/, add an EXPECTED
entry in test_scanner.py, confirm benign tests still pass.

safetensors specimens are synthesized in tests/test_safetensors.py (there is no
safetensors library here and no sample file on the box): `build()` writes a valid
file byte by byte, and each tamper test breaks exactly one invariant. Keep it that
way — a new structural check needs a specimen that breaks it and a benign case that
must stay clean.

## House rules

- Python stdlib + `gguf` + `huggingface_hub` only. No new deps without discussion.
  (`gguf` is imported lazily-on-failure, so safetensors audits work with no third-party
  package installed at all; keep it that way.)
- No generated artifacts in commits (reports/, baselines/, *.tensorhashes.json).
- Attribution: ExileAI. MIT license.
