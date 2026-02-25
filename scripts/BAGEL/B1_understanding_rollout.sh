#!/usr/bin/env bash
set -euo pipefail

# Backward-compatible wrapper.
# Canonical launcher now lives in Bagel/scripts:
#   bash Bagel/scripts/B1_unified_training.sh

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
TARGET="$REPO_ROOT/Bagel/scripts/B1_unified_training.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "[B1-wrapper] ERROR: target launcher not found: $TARGET" >&2
  exit 1
fi

exec "$TARGET" "$@"
