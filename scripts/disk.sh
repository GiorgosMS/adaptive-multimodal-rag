#!/usr/bin/env bash
source "$(dirname "${BASH_SOURCE[0]}")/env.sh" >/dev/null
echo "Everything below is safe to delete (rm -rf \"\$AMRAG_CACHE\"):"
du -sh "$AMRAG_CACHE"/* 2>/dev/null | sort -h
echo "---"
du -sh "$AMRAG_CACHE"
df -h --output=target,avail "$AMRAG_CACHE" | tail -1
