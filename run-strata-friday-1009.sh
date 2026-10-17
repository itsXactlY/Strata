#!/bin/sh
# Fri 2026-10-09 19:35 endpoint: engine 806442f (0.1.37 + TAPE) + strata-friday-1009.json (86k), server.py from this commit
here="$(cd "$(dirname "$0")" && pwd)"
exec "/home/alca/projects/llama.cpp-strata/.venv/bin/python" "$here/serve/server.py" "--engine" "strata" "--config" "$here/strata-friday-1009.json" "--port" "8080"
