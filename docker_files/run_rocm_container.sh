docker run -it \
  --name docplag-rocm \
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
  bash -lc "pip install --no-deps -r scripts/requirements-rocm.txt && bash"