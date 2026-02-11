#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT/src" python -m train --config configs/config.yaml "$@"
