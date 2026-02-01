#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/.."

PYTHON_BIN="python"
if [[ -x ".venv/bin/python" ]]; then
  PYTHON_BIN=".venv/bin/python"
elif [[ -x "venv/bin/python" ]]; then
  PYTHON_BIN="venv/bin/python"
fi

"$PYTHON_BIN" -m cot_bias fetch --update
"$PYTHON_BIN" -m cot_bias compute --out outputs
"$PYTHON_BIN" -m cot_bias dashboard --out outputs
"$PYTHON_BIN" scripts/generate_recent_dashboards.py --months 4 --out outputs

echo "Done. Latest output in ./outputs/ and last 4 months in ./outputs/YYYY-MM-DD/"
