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
one() { # one <dir> <out-basename> <relative-model-path...>
  local d="$1" name="$2"; shift 2
  mkdir -p "$WORK/$d"
  ( cd "$MODELS" && python3 "$REPO/report.py" -o "$WORK/$d" "$@" ) >/dev/null
  mv "$WORK/$d"/*.html "$HERE/$name.html"
  mv "$WORK/$d"/*.json "$HERE/$name.json"
}

# 01 — clean flagship (also the honest, properly-attributed case)
one 01 report-01-clean-gemma-12b-obliterated \
    "OBLITERATUS/Gemma-4-12B-OBLITERATED/Gemma-4-12B-OBLITERATED-v2-Q4_K_M.gguf"

# 02 — USE WITH AWARENESS (source + uploader handle kept out of this repo)
if [ -n "$WARN_SRC" ]; then
  mkdir -p "$WORK/warn-src"
  ln -sf "$WARN_SRC" "$WORK/warn-src/uncensored-20b-mxfp4.gguf"
  ( cd "$WORK/warn-src" && python3 "$REPO/report.py" -o "$WORK/02" "uncensored-20b-mxfp4.gguf" ) >/dev/null
  mv "$WORK/02"/*.html "$HERE/report-02-warn-mxfp4-anonymous.html"
  mv "$WORK/02"/*.json "$HERE/report-02-warn-mxfp4-anonymous.json"
  python3 - "$REDACT" "$HERE/report-02-warn-mxfp4-anonymous.html" "$HERE/report-02-warn-mxfp4-anonymous.json" <<'PY'
import sys, pathlib
tokens = [t for t in sys.argv[1].split() if t]
for f in sys.argv[2:]:
    p = pathlib.Path(f); t = p.read_text()
    for tok in tokens:
        t = t.replace(tok, "REDACTED")
    p.write_text(t)
PY
else
  echo "skip 02 (set WARN_SRC=/path.gguf REDACT=Token to regenerate the WARN sample)"
fi

# 03 — two uploaders, same AEON lineage (derivative-looking pair)
one 03 report-03-aeon-pair-two-uploaders \
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
( cd "$WORK/dnr" && python3 "$REPO/report.py" -o "$WORK/04" "Gemma-4-12B-OBLITERATED-demo-tampered-template.gguf" ) >/dev/null
mv "$WORK/04"/*.html "$HERE/report-04-do-not-run-template-swap.html"
mv "$WORK/04"/*.json "$HERE/report-04-do-not-run-template-swap.json"

echo "regenerated samples in $HERE"
if grep -rIl --include='*.html' --include='*.json' "/home/\|/neo" "$HERE"; then echo "LEAK — see above"; exit 1; else echo "leak check: clean"; fi
if [ -n "$REDACT" ] && grep -rIl --include='*.html' --include='*.json' "$REDACT" "$HERE"; then echo "REDACT token still present"; exit 1; fi
echo "ok"
