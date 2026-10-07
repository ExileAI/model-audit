# Safe workflow: from download to (justified) trust

This is the recipe `docs/limitations.md` implies. Follow it in order.

## 0. Before you download anything

Ask three questions about the uploader:
1. Do they document their method (abliteration layers, base model, quantization recipe)?
2. Do they publish in **safetensors** with a README explaining what they changed?
3. Do they have history, or is this a fresh account whose only uploads are GGUFs?

Two or three "no" answers → treat the file as untrusted no matter what any audit says.

## 1. At download: pin and record

Record three things in a note next to the model — the moment you download is the only
time you can pin history:

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

- The first command proves your copy is what the repo published at that commit.
  (It works the same for `model.safetensors` — the format is auto-detected.)
- If the repo ships its own hash list (`MANIFEST.txt`, `SHA256SUMS`, `*.sha256`), the
  audit checks your file against it and tells you if the file was renamed after hashing.
  A mismatch under the file's own name is a red flag.
- The second writes `model.gguf.tensorhashes.json` — the per-weight fingerprint.
  **Keep it next to the model, forever.** It is your evidence that the weights you
  run today are the weights you audited today.

Read the report. CRIT findings → do not load the model, full stop. Warnings → read
each explanation; tool-calling models trip "file I/O"-style patterns legitimately.

### 2b. If the model is safetensors, hash the sidecars too

A safetensors file carries no chat template: the instructions live in sidecars
(`chat_template.jinja`, `tokenizer_config.json`, `config.json`). Those are swappable
on their own, so hash them alongside the weights:

```bash
sha256sum model.safetensors *.jinja tokenizer_config.json config.json > SIDECARS.sha256
```

The audit already scans every one of those templates for hostile content — but the
hash is what lets you prove later that the file you read is still the file in place.

### 2c. If you have the source tree, prove the conversion

When a build ships both the fp/bf16 source and a converted GGUF, fingerprint both and
compare across formats:

```bash
python3 model_audit.py model.safetensors --tensor-hashes
python3 model_audit.py model-Q8_0.gguf   --tensor-hashes
python3 model_audit.py --diff model.safetensors.tensorhashes.json model-Q8_0.gguf.tensorhashes.json
```

- A **lossless** conversion matches on every blob: byte-identical where the container type
  is the same, and value-identical where the converter changed only the storage type (bf16
  norms widened to f32 keep their numbers exactly). That is the strongest statement this
  tool can make about provenance: the build carries the weights of a tree you audited.
- A **quantized** build matches only on the tensors quantization leaves untouched (usually
  the norms). The rest is reported as "not comparable this way" — quantization transforms
  values by design, so that is not a red flag.
- Names differ between formats (`blk.0.attn_q` vs `model.layers.0.self_attn.q_proj`) and so
  can dimension order; the comparison matches by content for exactly that reason.

## 3. Whenever you re-download or copy the model

Re-run both commands. The `--tensor-hashes` JSON now pays off:

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
- Re-check the repo occasionally: if the pinned revision's files change upstream,
  your copy is now the only honest one — and the uploader has some explaining to do.

## What this workflow still cannot do

- Prove weights are benign in isolation (see `docs/limitations.md`).
- Detect post-quantization training without a trusted baseline of the same base.
- Vouch for an uploader whose files hash perfectly.

The gold standard remains: documented safetensors source → you quantize yourself →
you own the hashes from birth.
