# What model-audit can and cannot tell you

Read this before trusting any audit verdict. This tool narrows uncertainty; it cannot
eliminate it.

## What the audit actually proves

| Check | Proves | Does NOT prove |
|---|---|---|
| SHA-256 (local) | This exact byte stream is what you hashed | That those bytes are safe |
| `--hf` remote check | Your copy matches what the repo publishes **at that moment/revision** | That the repo's copy was ever safe — you inherit whatever trust the uploader has |
| `--revision` pin | Match against a specific, immutable commit | Anything about commits *before* it; a repo can publish poison at v1 and swap at v2 |
| Template scan | The chat instructions carried inside the file contain no (detected) hostile code, hidden chars, or exfiltration patterns | That the scanner's pattern list is complete. Novel attacks evading these signatures will pass |
| Metadata checks | Internal labels agree with filename claims; provenance KVs exist | That the labels are truthful — an uploader can set `general.name` to anything |
| Tensor census | Structure is complete and sane (no missing/empty blocks) | That tensor *values* are what they should be |
| safetensors header checks | The header parses, its declared lengths fit the file, each tensor's dtype × shape equals its declared byte span, spans never overlap, and no bytes are unaccounted for (gap or trailing data) | That the weights are what the header says — the header is text the uploader wrote. A perfectly self-consistent file can be a perfectly tampered one |
| Template sidecar scan (safetensors) | The templates shipped next to the weights (`chat_template.jinja`, every `chat_template` entry in `tokenizer_config.json`) contain no (detected) hostile code, and two disagreeing sources are flagged | That those are the templates that will actually be used — your app may supply its own — or that the pattern list is complete |
| `*.index.json` check | The shard description matches the shard it describes (weights present, `total_size` right) | That the other shards in the set, or the tensors' contents, are honest |
| `auto_map` check | The uploader ships Python that loaders may execute when `trust_remote_code` is on | What that code does. Read it, or leave `trust_remote_code` off — that flag is the actual decision |
| `--tensor-hashes` | A stable per-block fingerprint for **comparison** | Anything, without a trusted baseline to compare against |

## The things this tool cannot do (and why)

1. **Prove weights are benign.** Weight-level backdoors live in the math, not the
   metadata. No scan of this kind detects a weight-tampered model without a clean
   reference of the same base.

2. **Detect post-quantization training (PQT) without a baseline.** If someone
   quantizes a model, then trains the quantized weights (or ablitrates *after*
   quantizing), the file looks structurally perfect. Detection requires comparing
   per-tensor hashes or statistical weight profiles against a known-pure quant of the
   same base. No baseline = no verdict.

3. **See upstream history that was never pinned.** If nobody recorded the commit SHA
   (or the pre-conversion safetensors state) at download time, a force-push is
   invisible in hindsight. HF is not git — history can simply vanish. This is why the
   workflow is: pin the commit **at download**, hash immediately, re-check against the
   pin later.

4. **Vouch for the uploader.** All checks pass on a perfectly tampered file published
   by a hostile account. Provenance KVs (`general.quantized_by` etc.) are
   self-reported. The strongest provenance remains: documented abliteration in
   safetensors + you run the quantization yourself.

5. **Cover a safetensors model's behavior with one hash.** In GGUF the chat template
   is inside the file you hashed, so the hash covers it. In safetensors it is not —
   the template is a sidecar, and the sidecar can be swapped without touching the
   weights at all. Hash the sidecars too (`--tensor-hashes` fingerprints the weights;
   a plain `sha256sum *.jinja tokenizer_config.json` covers the instructions), and
   re-check them on every re-download. A clean weights hash says nothing about the
   text your model reads its instructions from.

6. **Verify a quantized GGUF against its safetensors source.** Names and shapes can
   always be matched against the upstream source tree. Byte-for-byte tensor hashes can
   only be matched for *lossless* conversions (F16/F32 style) — block-quantized weights
   (Q4_K_M and friends) are numerically transformed, so their hashes will differ by
   design. `--diff` warns when you compare baselines made from different formats; do not
   read those differences as tampering.

## What strong verification actually looks like

The gold standard, in order:

1. Source repo documents its abliteration method (layers touched, refusal direction,
   base model), in **safetensors**. Audit that source tree with this tool
   (`python3 model_audit.py model.safetensors` — it scans every template sidecar next
   to the file) and fingerprint it with `--tensor-hashes`.
2. You quantize yourself with your own llama.cpp build → you own the GGUF and its
   hashes from birth, and you know exactly which source produced it.
3. If you must take a prebuilt GGUF: trusted long-history quantizer, `--hf` + pinned
   commit at download, `--tensor-hashes` saved alongside, re-check on any re-download.

## Practical guidance from verdicts

- `DO NOT RUN` — hostile template or structural breakage found. Don't load it.
- `USE WITH AWARENESS` — read the specific warnings. Tool-calling models trip
  "file I/O" style patterns legitimately; the explanation text tells you when.
- `NO RED FLAGS` — the file is honest about what it is, **not** that it is safe.
  For anything you care about, generate `--tensor-hashes` and keep them.
