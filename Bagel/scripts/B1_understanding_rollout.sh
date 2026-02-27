#!/usr/bin/env bash
set -euo pipefail

# Compatibility wrapper retained for older commands.
# Unified BAGEL training is handled by B1_unified_training.sh.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET="$SCRIPT_DIR/B1_unified_training.sh"

if [[ ! -f "$TARGET" ]]; then
  echo "[B1] ERROR: Missing target script: $TARGET" >&2
  exit 1
fi

bash "$TARGET" "$@"

