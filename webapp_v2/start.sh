#!/usr/bin/env bash
# ============================================================
#  Gauge OCR - Web UI one-click launcher (Linux / macOS)
#  Opens http://127.0.0.1:8770/ ; Ctrl+C to stop.
# ============================================================
set -euo pipefail
DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PORT="${PORT:-8770}"

# 两步都是 YOLO，需要 ultralytics(+torch) 与 opencv：
# 优先用本机 conda 的 wanhua 环境，只有它不存在时才退回 PATH 上的 python
# （那个环境不一定装了 ultralytics）。
PY=""
if [ -x "/e/Anaconda/envs/wanhua/python.exe" ]; then
  PY="/e/Anaconda/envs/wanhua/python.exe"      # Git Bash on Windows
elif [ -x "/mnt/e/Anaconda/envs/wanhua/python.exe" ]; then
  PY="/mnt/e/Anaconda/envs/wanhua/python.exe"  # WSL on Windows
else
  PY="$(command -v python3 || command -v python || true)"
fi
if [ -z "$PY" ]; then
  echo "[ERROR] python not found in PATH" >&2
  exit 1
fi

if ! "$PY" -c "import torch, cv2, ultralytics" >/dev/null 2>&1; then
  echo "[ERROR] ultralytics / torch / cv2 missing in $PY" >&2
  echo "        run: $PY -m pip install -r $DIR/requirements.txt" >&2
  exit 1
fi

echo "[1/2] Python : $PY"
echo "[2/2] Server : $DIR/server.py   port $PORT"
echo
exec "$PY" "$DIR/server.py" --port "$PORT" --open
