# What gguf-audit can and cannot tell you

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

## What strong verification actually looks like

The gold standard, in order:

1. Source repo documents its abliteration method (layers touched, refusal direction,
   base model), in **safetensors**.
2. You quantize yourself with your own llama.cpp build → you own the GGUF and its
   hashes from birth.
3. If you must take a prebuilt GGUF: trusted long-history quantizer, `--hf` + pinned
   commit at download, `--tensor-hashes` saved alongside, re-check on any re-download.

## Practical guidance from verdicts

- `DO NOT RUN` — hostile template or structural breakage found. Don't load it.
- `USE WITH AWARENESS` — read the specific warnings. Tool-calling models trip
  "file I/O" style patterns legitimately; the explanation text tells you when.
- `NO RED FLAGS` — the file is honest about what it is, **not** that it is safe.
  For anything you care about, generate `--tensor-hashes` and keep them.
