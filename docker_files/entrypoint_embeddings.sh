#!/usr/bin/env bash
set -euo pipefail

for path in /usr/lib/libdxcore.so /usr/lib/librocdxg.so; do
  if [ ! -f "$path" ]; then
    echo "ERROR: Expected ROCm/WSL GPU library file missing: $path" >&2
    echo "If this path is a directory, Docker likely created it because Compose was run from Windows instead of WSL." >&2
    echo "Run docker compose from WSL where /opt/rocm/lib/librocdxg.so exists, or switch the Streamlit embeddings backend to Local Python/cached embeddings for testing." >&2
    exit 78
  fi
done

exec "$@"
