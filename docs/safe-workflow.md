# Safe workflow: from download to (justified) trust

This is the recipe `docs/limitations.md` implies. Follow it in order.

## 0. Before you download anything

Ask three questions about the uploader:
1. Do they document their method (abliteration layers, base model, quantization recipe)?
2. Do they publish in **safetensors** with a README explaining what they changed?
3. Do they have history, or is this a fresh account whose only uploads are GGUFs?

Two or three "no" answers → treat the file as untrusted no matter what any audit says.

## 1. At download: pin and record

Record three things in a note next to the model — recording the resolved commit at download preserves the
snapshot you intended to compare:

```
repo:     <uploader>/<model-name>
revision: <commit SHA from the HF page, "Files and versions" → click the commit>
date:     <today>
```

## 2. Immediately after download: audit and baseline

```bash
python3 model_audit.py model.gguf --hf <uploader>/<repo> --revision <commit_sha>
python3 model_audit.py model.gguf --tensor-hashes
```

- A successful first check establishes a byte match to the identified remote file
  at that commit; a failed or unavailable lookup does not.
  (It works the same for `model.safetensors` — the format is auto-detected.)
- If the repo ships its own hash list (`MANIFEST.txt`, `SHA256SUMS`, `*.sha256`), the
  audit checks your file against it and tells you if the file was renamed after hashing.
  A mismatch under the file's own name is a red flag.
- The second writes `model.gguf.tensorhashes.json` — the per-weight fingerprint.
  **Keep that original reference.** Existing destinations are refused rather than
  overwritten, with WARN and exit 1 (exit 2 if another finding is CRIT). Choose a
  new destination with `--baseline-out`. This fingerprints the audited bytes; it
  does not prove the weights safe.

Read the report. `CRITICAL FINDINGS: REVIEW BEFORE LOADING` means investigate before
loading. `WARNINGS: REVIEW THE DETAILS` means read each explanation; tool-calling
models trip "file I/O"-style patterns legitimately. `NO RED FLAGS IN COMPLETED CHECKS`
does not cover anything absent or uninspected and does not promise safety.
An empty safetensors tensor gets an inventory-heuristic WARN but remains
structurally legal and baseline-eligible.

### 2b. If the model is safetensors, hash the sidecars too

A safetensors file carries no chat template: the instructions live in sidecars
(`chat_template.jinja`, `tokenizer_config.json`, `config.json`). Those are swappable
on their own, so hash them alongside the weights:

```bash
sha256sum model.safetensors *.jinja tokenizer_config.json config.json > SIDECARS.sha256
```

The audit scans supported template representations for known suspicious patterns — but the
hash is what lets you prove later that the file you read is still the file in place.

### 2c. Compare source and converted fingerprint evidence

When a build ships both the fp/bf16 source and a converted GGUF, fingerprint both and
compare across formats:

```bash
python3 model_audit.py model.safetensors --tensor-hashes
python3 model_audit.py model-Q8_0.gguf   --tensor-hashes
python3 model_audit.py --diff model.safetensors.tensorhashes.json model-Q8_0.gguf.tensorhashes.json
```

- Matching proceeds by bytes first, supported typed normalized fingerprints second,
  then ambiguous candidates. The counts stay separate; each source and destination
  tensor is consumed at most once. Both unmatched sides have complete counts and
  up to eight names each.
- New baselines preserve `tensors`, `values`, and `shapes` and add optional per-tensor
  `value_metadata` (`dtype`, `byte_order: "little"`, `normalization: "f32-le-bits-v1"`).
  Little-endian BF16/F32 tensors have explicit normalized `values` hashes,
  including F32. Only value-to-value matches with supported metadata on both sides
  establish an **exact normalized-F32-fingerprint match**. Big- or unknown-endian GGUF
  tensors get byte hashes only, without value claims.
- F16's existing struct-based normalized hash is only a legacy/inconclusive
  candidate because it does not preserve NaN payload bits. Baseline validation
  rejects typed F32 metadata whose byte and normalized hashes contradict each other.
- BF16 widening to F32 is mathematically exact. Matching fingerprints establish
  consistency of the normalized bitstream, not historical lineage or a lossless
  whole-model conversion. A known container format cannot prove a raw hash's dtype:
  an old BF16 value hash can match I32 raw bits. Legacy/ambiguous candidates remain
  INCONCLUSIVE and get WARN findings.
- Cross-format PASS requires nonempty, full byte/typed-normalized correspondence.
  It does not check shapes, semantic roles, numerical/model equivalence, or history.
  Unknown formats, ambiguous candidates, empty inventories, and partial matches
  remain CONCERN. Empty inventories receive WARN; general comparison limits are INFO.
- A **quantized** build typically matches on tensors left unquantized (usually the
  norms). The rest is reported as "not comparable this way" — quantization transforms
  values by design, so nonmatches are INFO and a comparison CONCERN, not evidence
  of tampering. A CONCERN may therefore accompany exit 0; read the rubric too.
- Names differ between formats (`blk.0.attn_q` vs `model.layers.0.self_attn.q_proj`) and so
  can dimension order; the comparison matches by content for exactly that reason.

## 3. Whenever you re-download or copy the model

Preserve the original baseline and write a distinct new one:

```bash
python3 model_audit.py model.gguf --hf <uploader>/<repo> --revision <original_commit_sha>
python3 model_audit.py model.gguf --tensor-hashes --baseline-out model.new.json
python3 model_audit.py --diff model.gguf.tensorhashes.json model.new.json
```

Choose a fresh destination for each snapshot; neither output replaces your reference.
Finding exit codes are expected when differences or warnings are present. Read the
findings rather than treating exit 1/2 as successful verification.

- Same file SHA-256 → identical copy.
- Different SHA-256 but you need to know *what* changed → compare tensor baselines
  (see `diff` in the README): metadata-only change vs. swapped weight blocks.

## 4. Ongoing hygiene

- Never run scripts, converters, or "helpers" shipped in the same repo/folder as a
  GGUF. The weights are the artifact; sidecars are a separate supply chain.
- Prefer loading the template baked into the GGUF; if your app offers to use an
  external `chat_template.jinja`, audit that file too (it overrides the in-file one).
  For safetensors there is nothing baked in — the sidecar *is* the template, so audit
  and hash it.
- Never enable `trust_remote_code` for a model you have not read. If the audit reports
  `auto_map` in `config.json` or `tokenizer_config.json`, the repo ships Python that
  your loader will execute — that is code you are running, not weights you are loading.
- Retain the full resolved upstream commit. A mismatch or unavailable reference
  needs investigation; it alone does not prove a force-push, malicious replacement,
  or that either copy is honest.

## What this workflow still cannot do

- Prove weights are benign in isolation (see `docs/limitations.md`).
- Detect post-quantization training without a trusted baseline of the same base.
- Vouch for an uploader whose files hash perfectly.

The gold standard remains: documented safetensors source → you quantize yourself →
you own the hashes from birth.
