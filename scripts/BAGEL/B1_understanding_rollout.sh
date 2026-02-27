#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper: this path was used earlier in runs.
# It now delegates to the unified BAGEL experiment launcher.

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/B1_unified_training.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "[B1] ERROR: Missing target script: $TARGET" >&2
  exit 1
fi

bash "$TARGET" "$@"

