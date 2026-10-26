#!/usr/bin/env bash
set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTRUCTION_CONDITION=explicit_spatial exec bash "$SCRIPT_DIR/eval_attention_all.sh"
