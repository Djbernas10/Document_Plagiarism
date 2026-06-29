#!/usr/bin/env bash
set -euo pipefail

for path in /usr/lib/wsl/lib/libdxcore.so /opt/rocm/lib/librocdxg.so; do
  if [ ! -f "$path" ]; then
    echo "ERROR: Expected ROCm/WSL file does not exist or is not a file: $path" >&2
    echo "Current state:" >&2
    ls -ld "$path" 2>/dev/null || true
    echo >&2
    echo "Run this script from WSL, and make sure the path above is a real file." >&2
    echo "If it is a directory, remove that bad directory and restore the ROCm file/path." >&2
    exit 78
  fi
done

docker rm -f docplag-rocm-embeddings >/dev/null 2>&1 || true

docker run -it \
  --name docplag-rocm-embeddings \
  -p 8000:8000 \
  -v /mnt/c/projects/thesis/Document_Plagiarism:/workspace/Document_Plagiarism \
  -w /workspace/Document_Plagiarism \
  -v /usr/lib/wsl/lib/libdxcore.so:/usr/lib/libdxcore.so \
  -v /opt/rocm/lib/librocdxg.so:/usr/lib/librocdxg.so \
  -e HSA_ENABLE_DXG_DETECTION=1 \
  --device=/dev/dxg \
  --cap-add=SYS_PTRACE \
  --security-opt seccomp=unconfined \
  --ipc=host \
  --shm-size 12G \
  rocm/pytorch:latest \
  bash -lc "pip install --no-deps -r scripts/requirements-rocm.txt && python scripts/embedding_service.py"
