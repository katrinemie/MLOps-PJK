#!/bin/bash
ROOT="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT"
PYTHONPATH="$ROOT/src" python -m inference --model models/best_model.pt "$@"
