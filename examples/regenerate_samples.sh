#!/usr/bin/env bash
# Regenerate the sample reports in this directory.
#
#   MODELS=~/.lmstudio/models  ./regenerate_samples.sh
#
# Every sample is a real audit of a real file, except report-04, whose file is
# a clean model with ONLY its chat template swapped for a hostile specimen
# (see synthesize_do_not_run.py). Output carries no personal paths: audits run
# with relative paths, so a home directory is never printed.
#
# The WARN sample (02) is deliberately unattributed, so its source file and the
# token to scrub are supplied at run time and are NOT recorded here:
#
#   WARN_SRC=/path/to/a-file-that-warns.gguf REDACT="UploaderHandle" \
#       ./regenerate_samples.sh
#
# If WARN_SRC is unset, sample 02 is skipped (the other three still regenerate).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"
REPO="$(dirname "$HERE")"
MODELS="${MODELS:-$HOME/.lmstudio/models}"
WARN_SRC="${WARN_SRC:-}"
REDACT="${REDACT:-}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT
# Finding statuses are expected for the WARN/CRIT examples. Accept only the
# requested status and a complete artifact pair; never hide execution failures.
collect() { # collect <expected-status> <cwd> <dir> <out-basename> <model...>
  local expected="$1" cwd="$2" d="$3" name="$4"; shift 4
  local html status=0 json
  mkdir -p "$WORK/$d"
  html="$(cd "$cwd" && python3 "$REPO/report.py" -o "$WORK/$d" "$@")" || status=$?
  if [ "$status" -ne "$expected" ]; then
    echo "unexpected report status $status (expected $expected): $name" >&2
    return 1
  fi
  json="${html%.html}.json"
  if [[ "$html" != "$WORK/$d/"* || "$html" != *.html || ! -f "$html" || ! -f "$json" ]]; then
    echo "report did not produce a complete pair: $name" >&2
    return 1
  fi
  # A deliberately critical sample must contain the intended template finding,
  # not merely a missing-file/parser failure that also returns exit 2.
  python3 - "$json" "$expected" <<'CHECK'
import json, sys
with open(sys.argv[1], encoding="utf-8") as f:
    reports = json.load(f)
items = [item for report in reports for item in report["findings"]]
if any(item["section"] == "audit" and item["severity"] == "CRIT" for item in items):
    raise SystemExit("sample audit failed")
if sys.argv[2] == "2" and not any(item["section"] == "template" and item["severity"] == "CRIT" for item in items):
    raise SystemExit("expected critical template specimen was not detected")
CHECK
  mv "$html" "$HERE/$name.html"
  mv "$json" "$HERE/$name.json"
}
one() { # one <expected-status> <dir> <out-basename> <relative-model-path...>
  local expected="$1" d="$2" name="$3"; shift 3
  collect "$expected" "$MODELS" "$d" "$name" "$@"
}

# 01 — clean flagship (also the honest, properly-attributed case)
one 0 01 report-01-clean-gemma-12b-obliterated \
    "OBLITERATUS/Gemma-4-12B-OBLITERATED/Gemma-4-12B-OBLITERATED-v2-Q4_K_M.gguf"

# 02 — USE WITH AWARENESS (source + uploader handle kept out of this repo)
if [ -n "$WARN_SRC" ]; then
  mkdir -p "$WORK/warn-src"
  ln -sf "$WARN_SRC" "$WORK/warn-src/uncensored-20b-mxfp4.gguf"
  collect 1 "$WORK/warn-src" 02 report-02-warn-mxfp4-anonymous "uncensored-20b-mxfp4.gguf"
  python3 - "$REDACT" "$HERE/report-02-warn-mxfp4-anonymous.html" "$HERE/report-02-warn-mxfp4-anonymous.json" <<'PY'
import sys, pathlib
tokens = [t for t in sys.argv[1].split() if t]
for f in sys.argv[2:]:
    p = pathlib.Path(f); t = p.read_text(encoding="utf-8")
    for tok in tokens:
        t = t.replace(tok, "REDACTED")
    p.write_text(t, encoding="utf-8")
PY
else
  echo "skip 02 (set WARN_SRC=/path.gguf REDACT=Token to regenerate the WARN sample)"
fi

# 03 — two uploaders, same AEON lineage (derivative-looking pair)
one 0 03 report-03-aeon-pair-two-uploaders \
    "vcruz305/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-GGUF/Qwen3.8-27B-AEON-ULTIMATE-UNCENSORED-Q4_K_M.gguf" \
    "Abiray/Qwen3.6-27B-AEON-Ultimate-Uncensored-GGUF/qwen3.6-27b-uncensored-q4_k_s.gguf"

# 04 — DO NOT RUN: a clean model with only its chat template swapped
SRCDIR="$MODELS/OBLITERATUS/Gemma-4-12B-OBLITERATED"
mkdir -p "$WORK/dnr"
python3 "$HERE/synthesize_do_not_run.py" \
    "$SRCDIR/Gemma-4-12B-OBLITERATED-v2-Q4_K_M.gguf" \
    "$REPO/templates/tampered/03-invisible-instructions.j2" \
    "$WORK/dnr/Gemma-4-12B-OBLITERATED-demo-tampered-template.gguf"
cp -n "$SRCDIR"/*.json "$WORK/dnr/" 2>/dev/null || true
collect 2 "$WORK/dnr" 04 report-04-do-not-run-template-swap "Gemma-4-12B-OBLITERATED-demo-tampered-template.gguf"

echo "regenerated samples in $HERE"
if grep -rIl --include='*.html' --include='*.json' "/home/\|/neo" "$HERE"; then echo "LEAK — see above"; exit 1; else echo "leak check: clean"; fi
if [ -n "$REDACT" ] && grep -rIl --include='*.html' --include='*.json' "$REDACT" "$HERE"; then echo "REDACT token still present"; exit 1; fi
echo "ok"
