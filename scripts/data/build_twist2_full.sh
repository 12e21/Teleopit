#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SPEC="train_mimic/configs/datasets/twist2_full.yaml"
FORCE=0
SKIP_FK_CHECK=0
JOBS=1

usage() {
  cat <<USAGE
Usage:
  bash scripts/data/build_twist2_full.sh [options]

Options:
  --force            Delete twist2_full cache/build outputs before rebuild
  --skip-fk-check    Skip sampled FK checks
  --jobs <n>         Source-level conversion jobs (default: 1)
  -h, --help         Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --force)
      FORCE=1
      shift
      ;;
    --skip-fk-check)
      SKIP_FK_CHECK=1
      shift
      ;;
    --jobs)
      JOBS="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown argument: $1" >&2
      usage >&2
      exit 1
      ;;
  esac
done

cmd=(python scripts/data/build_dataset_v2.py --spec "$SPEC" --jobs "$JOBS")
if [[ "$FORCE" == "1" ]]; then
  cmd+=(--force)
fi
if [[ "$SKIP_FK_CHECK" == "1" ]]; then
  cmd+=(--skip_fk_check)
fi

"${cmd[@]}"
