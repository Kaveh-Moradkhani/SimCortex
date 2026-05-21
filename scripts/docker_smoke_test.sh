#!/usr/bin/env bash
set -euo pipefail

IMAGE="${1:-simcortex:2.0.0}"

echo "==> Smoke test for image: ${IMAGE}"

echo
echo "==> [1/5] CLI root help"
docker run --rm "${IMAGE}" simcortex --help >/dev/null

echo
echo "==> [2/5] Segmentation CLI help"
docker run --rm "${IMAGE}" simcortex seg --help >/dev/null

echo
echo "==> [3/5] InitSurf CLI help"
docker run --rm "${IMAGE}" simcortex initsurf --help >/dev/null

echo
echo "==> [4/5] Deform CLI help"
docker run --rm "${IMAGE}" simcortex deform --help >/dev/null

echo
echo "==> [5/5] Python imports + CUDA"
docker run --rm --gpus all "${IMAGE}" python - <<'PY'
import torch
import pytorch3d
import pymeshlab

print("torch:", torch.__version__)
print("cuda_available:", torch.cuda.is_available())
print("device_count:", torch.cuda.device_count())

if not torch.cuda.is_available():
    raise SystemExit("CUDA is not available inside container")
PY

echo
echo "==> All smoke tests passed."
