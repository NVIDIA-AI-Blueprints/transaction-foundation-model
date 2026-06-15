#!/usr/bin/env bash
set -euo pipefail

ROOT="$(git rev-parse --show-toplevel 2>/dev/null || pwd)"

echo "scripts/gcp-jupyter.sh is deprecated; starting marimo via scripts/gcp-marimo.sh instead." >&2
exec "$ROOT/scripts/gcp-marimo.sh" "$@"
