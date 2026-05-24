#!/usr/bin/env bash
# Health check and startup hints for Harness MVP microservices.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
QWEN_DIR="${QWEN_DIR:-/Applications/School/Qwen-VL-master}"
TRUFOR_DIR="${TRUFOR_DIR:-/Applications/School/computer/temporary/TruFor/test_docker}"

QWEN_URL="${QWEN_URL:-http://127.0.0.1:8000}"
TRUFOR_URL="${TRUFOR_URL:-http://127.0.0.1:8001}"
RAG_URL="${RAG_URL:-http://127.0.0.1:8002}"

check_url() {
  local name="$1"
  local url="$2"
  if curl -sf "$url" >/dev/null 2>&1; then
    echo "  [OK]   $name ($url)"
    return 0
  else
    echo "  [FAIL] $name ($url)"
    return 1
  fi
}

echo "=== Harness MVP Service Health Check ==="
echo

qwen_ok=0
trufor_ok=0
rag_ok=0

check_url "Qwen"   "$QWEN_URL/health" && qwen_ok=1 || true
# TruFor has no /health; POST without file returns 422/400 when alive
if curl -sf -o /dev/null -w "%{http_code}" -X POST "$TRUFOR_URL/score" 2>/dev/null | grep -qE '^(400|422|503)$'; then
  echo "  [OK]   TruFor ($TRUFOR_URL)"
  trufor_ok=1
else
  echo "  [FAIL] TruFor ($TRUFOR_URL)"
fi
check_url "Mock RAG" "$RAG_URL/health" && rag_ok=1 || true

echo
if [[ "$qwen_ok" -eq 1 && "$trufor_ok" -eq 1 && "$rag_ok" -eq 1 ]]; then
  echo "All services are ready."
  exit 0
fi

echo "Some services are not running. Start them in separate terminals:"
echo
echo "# Terminal 1 — Mock RAG (port 8002)"
echo "cd \"$ROOT\" && uvicorn mocks.rag_server:app --host 127.0.0.1 --port 8002"
echo
echo "# Terminal 2 — Qwen3-VL-2B (port 8000)"
echo "cd \"$QWEN_DIR\""
echo "# Download weights first if missing:"
echo "# huggingface-cli download Qwen/Qwen3-VL-2B-Instruct --local-dir ./Qwen3-VL-2B"
echo "QWEN_CPU=true uvicorn app.main:app --host 127.0.0.1 --port 8000"
echo
echo "# Terminal 3 — TruFor (port 8001)"
echo "cd \"$TRUFOR_DIR/src\""
echo "# Download weights first if missing:"
echo "cd \"$TRUFOR_DIR\" && bash docker_build.sh  # or manually download to weights/"
echo "TRUFOR_DEVICE=cpu uvicorn api_server:app --host 127.0.0.1 --port 8001"
echo
exit 1
