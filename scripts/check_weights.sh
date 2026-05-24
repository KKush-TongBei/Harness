#!/usr/bin/env bash
# Check model weights for Qwen and TruFor before starting services.
set -euo pipefail

QWEN_DIR="${QWEN_DIR:-/Applications/School/Qwen-VL-master}"
TRUFOR_DIR="${TRUFOR_DIR:-/Applications/School/computer/temporary/TruFor/test_docker}"

echo "=== Model Weights Check ==="
echo

qwen_weights="${QWEN_DIR}/Qwen3-VL-2B"
if compgen -G "${qwen_weights}/*.safetensors" >/dev/null || compgen -G "${qwen_weights}/*.bin" >/dev/null; then
  echo "  [OK]   Qwen weights found in ${qwen_weights}"
else
  echo "  [MISS] Qwen weights NOT found in ${qwen_weights}"
  echo "         Run: huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ${qwen_weights}"
fi

trufor_weights="${TRUFOR_DIR}/weights/trufor.pth.tar"
if [[ -f "$trufor_weights" ]]; then
  echo "  [OK]   TruFor weights found: ${trufor_weights}"
else
  echo "  [MISS] TruFor weights NOT found: ${trufor_weights}"
  echo "         Run: cd ${TRUFOR_DIR} && bash docker_build.sh"
fi

echo
