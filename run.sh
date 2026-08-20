#!/usr/bin/env bash
# Launch Jtech-CLI. Pass extra args through, e.g. ./run.sh --no-discover
set -euo pipefail
cd "$(dirname "$0")"
exec uv run python -m jtech_cli "$@"
