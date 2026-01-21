#!/usr/bin/env bash
set -euo pipefail

IMAGE="mlagents-pipeline"

# Repo root = parent of this script's folder (scripts/)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

docker build -t "$IMAGE" "$ROOT"

docker run --rm -it --init \
  --user "$(id -u):$(id -g)" \
  -e USER="$(id -un)" -e LOGNAME="$(id -un)" \
  -v "$ROOT/training_manager/experiments/configs_for_training:/app/training_manager/experiments/configs_for_training" \
  -v "$ROOT/training_manager/experiments/results:/app/training_manager/experiments/results" \
  "$IMAGE" \
  python -m training_pipeline.cli.run_experiment "$@"
