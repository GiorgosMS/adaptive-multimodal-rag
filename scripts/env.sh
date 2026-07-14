#!/usr/bin/env bash
# Source before any work.  /  has ~35 GB free; the project drive has ~407 GB.
#
# EVERY downloaded byte lands under a single folder: $AMRAG_CACHE.
#   rm -rf "$AMRAG_CACHE"     <- reclaims all of it; the repo still works.
# Nothing else in the tree grows. Do not let any tool default to ~/.cache.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

export AMRAG_CACHE="$PROJECT_ROOT/_cache"
export HF_HOME="$AMRAG_CACHE/huggingface"       # models + datasets
export TORCH_HOME="$AMRAG_CACHE/torch"
export PIP_CACHE_DIR="$AMRAG_CACHE/pip"         # torch wheels are GBs
export AMRAG_DATA="$AMRAG_CACHE/data"           # our own derived artefacts

mkdir -p "$HF_HOME" "$TORCH_HOME" "$PIP_CACHE_DIR" "$AMRAG_DATA" || {
  echo "FATAL: cannot create $AMRAG_CACHE" >&2; return 1 2>/dev/null || exit 1; }

# The project path contains a space ("Personal Projects"). Fail loudly now
# rather than inside a subprocess that forgot to quote.
[ -w "$HF_HOME" ] || { echo "FATAL: $HF_HOME not writable" >&2; return 1 2>/dev/null || exit 1; }

echo "AMRAG_CACHE=$AMRAG_CACHE"
