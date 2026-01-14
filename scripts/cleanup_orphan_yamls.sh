#!/usr/bin/env bash
# cleanup_orphan_yamls.sh
#
# Looks for YAMLs named: <algo>_<hexhash>.yaml  (e.g. ppo_0a0b7caa37.yaml)
# Keeps YAML if ANY run folder exists under RESULTS_ROOT that contains the token <algo>_<hexhash>.
#
# Default action: DRY-RUN.
# Use --yes to actually do something.
# Default operation: MOVE orphans to a quarantine folder inside YAML_DIR: _orphaned/
# Use --delete to delete instead of move.

set -euo pipefail

YAML_DIR="training_manager/experiments/configs_for_training"
RESULTS_ROOT="training_manager/experiments/results"
DO_IT=0
DO_DELETE=0

usage() {
  cat <<EOF
Usage:
  $0 [--yaml-dir PATH] [--results-root PATH] [--yes] [--dry-run] [--delete]

Defaults:
  --dry-run (no changes)
  move orphans to: <yaml-dir>/_orphaned/

Examples:
  $0 --dry-run
  $0 --yes
  $0 --yes --delete
  $0 --yaml-dir training_manager/experiments/configs_for_training --results-root training_manager/experiments/results --yes
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --yaml-dir)     YAML_DIR="$2"; shift 2;;
    --results-root) RESULTS_ROOT="$2"; shift 2;;
    --yes)          DO_IT=1; shift;;
    --dry-run)      DO_IT=0; shift;;
    --delete)       DO_DELETE=1; shift;;
    -h|--help)      usage; exit 0;;
    *) echo "Unknown arg: $1" >&2; usage; exit 2;;
  esac
done

if [[ ! -d "$YAML_DIR" ]]; then
  echo "YAML dir not found: $YAML_DIR" >&2
  exit 1
fi
if [[ ! -d "$RESULTS_ROOT" ]]; then
  echo "Results root not found: $RESULTS_ROOT" >&2
  exit 1
fi

ORPHAN_DIR="${YAML_DIR}/_orphaned"

echo "YAML_DIR     : $YAML_DIR"
echo "RESULTS_ROOT : $RESULTS_ROOT"
echo "MODE         : $([[ $DO_IT -eq 1 ]] && echo APPLY || echo DRY-RUN)"
echo "ACTION       : $([[ $DO_DELETE -eq 1 ]] && echo DELETE || echo MOVE\ to\ $ORPHAN_DIR)"
echo

# 1) Index tokens from all run folders (depth 2: results/<machine>/<run>)
declare -A TOKENS=()

while IFS= read -r -d '' d; do
  base="$(basename "$d")"

  # Find a token like algo_hexhash anywhere in the folder name
  # Example: pawel_Windows_ppo_1f5abc123 -> matches ppo_1f5abc123
  if [[ "$base" =~ ([A-Za-z0-9]+)_([0-9a-fA-F]{6,}) ]]; then
    TOKENS["${BASH_REMATCH[1]}_${BASH_REMATCH[2]}"]=1
  fi
done < <(find "$RESULTS_ROOT" -mindepth 2 -maxdepth 2 -type d -print0)

echo "Indexed tokens from all results: ${#TOKENS[@]}"
echo "Scanning YAMLs..."
echo

kept=0
orphans=0
skipped=0
seen=0

# Ensure orphan dir exists if we will move
if [[ $DO_IT -eq 1 && $DO_DELETE -eq 0 ]]; then
  mkdir -p "$ORPHAN_DIR"
fi

while IFS= read -r -d '' yaml; do
  ((seen+=1))

  file="$(basename "$yaml")"
  if [[ ! "$file" =~ ^([A-Za-z0-9]+)_([0-9a-fA-F]{6,})\.yaml$ ]]; then
    ((skipped+=1))
    continue
  fi

  token="${BASH_REMATCH[1]}_${BASH_REMATCH[2]}"

  if [[ -n "${TOKENS[$token]+x}" ]]; then
    ((kept+=1))
  else
    ((orphans+=1))
    if [[ $DO_IT -eq 1 ]]; then
      if [[ $DO_DELETE -eq 1 ]]; then
        rm -f -- "$yaml"
      else
        mv -n -- "$yaml" "$ORPHAN_DIR/"
      fi
    fi
  fi
done < <(find "$YAML_DIR" -maxdepth 1 -type f -name "*.yaml" -print0)

echo
echo "Kept                  : $kept"
echo "Orphans (moved/deleted): $orphans"
echo "Skipped (nonmatch)    : $skipped"
echo "Total YAMLs seen      : $seen"
echo "Done."
