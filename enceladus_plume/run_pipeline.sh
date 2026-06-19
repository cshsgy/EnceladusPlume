#!/usr/bin/env bash
# Run the default parameter-sweep pipeline.
#
# Uses the currently active Python environment. If a local virtualenv exists
# at ./.venv it is activated automatically; otherwise the active interpreter
# is used as-is.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"
if [[ -f .venv/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv/bin/activate
fi
python run_pipeline.py --config config/run_config.yaml -v "$@"
