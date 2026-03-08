#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

SOURCE_ROOT="data/twist2_retarget_pkl"
MANIFEST="data/motion/manifests/twist2_full_v1.csv"
DATASET_VERSION="twist2_full_v1_30hz"
BUILD_ROOT="data/motion/builds"
NPZ_ROOT="."
NPZ_CLIPS_ROOT="data/motion/npz_clips"
TARGET_FPS=30
VAL_PERCENT=5
HASH_SALT=""
FORCE_CLEAN=0

SOURCES=(
  "OMOMO_g1_GMR"
  "AMASS_g1_GMR8"
  "twist1_to_twist2"
  "v1_v2_v3_g1"
)

usage() {
  cat <<USAGE
Usage:
  bash scripts/data/build_twist2_full.sh [options]

Options:
  --source-root <dir>       PKL root (default: ${SOURCE_ROOT})
  --manifest <path>         Manifest output path (default: ${MANIFEST})
  --dataset-version <name>  Build version (default: ${DATASET_VERSION})
  --target-fps <int>        Merge target fps (default: ${TARGET_FPS})
  --val-percent <int>       Validation split percent for empty split (default: ${VAL_PERCENT})
  --hash-salt <text>        Optional hash split salt
  --force-clean             Delete manifest/build dir for this version before rebuilding
  -h, --help                Show this help

Behavior:
  - Scans these PKL sources under --source-root if they exist:
      ${SOURCES[*]}
  - Ingests all found PKL files into data/motion/npz_clips/<source>/...
  - Writes manifest to ${MANIFEST}
  - Builds merged train/val under data/motion/builds/<dataset-version>/

Example:
  bash scripts/data/build_twist2_full.sh
  bash scripts/data/build_twist2_full.sh --force-clean
  bash scripts/data/build_twist2_full.sh --source-root /data/twist2_retarget_pkl
USAGE
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --source-root)
      SOURCE_ROOT="$2"
      shift 2
      ;;
    --manifest)
      MANIFEST="$2"
      shift 2
      ;;
    --dataset-version)
      DATASET_VERSION="$2"
      shift 2
      ;;
    --target-fps)
      TARGET_FPS="$2"
      shift 2
      ;;
    --val-percent)
      VAL_PERCENT="$2"
      shift 2
      ;;
    --hash-salt)
      HASH_SALT="$2"
      shift 2
      ;;
    --force-clean)
      FORCE_CLEAN=1
      shift
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

if ! [[ "$TARGET_FPS" =~ ^[0-9]+$ ]] || [[ "$TARGET_FPS" -le 0 ]]; then
  echo "ERROR: --target-fps must be a positive integer" >&2
  exit 1
fi

if ! [[ "$VAL_PERCENT" =~ ^[0-9]+$ ]] || [[ "$VAL_PERCENT" -le 0 || "$VAL_PERCENT" -ge 100 ]]; then
  echo "ERROR: --val-percent must be in [1, 99]" >&2
  exit 1
fi

SOURCE_ROOT="$(python - <<PY
from pathlib import Path
print(Path(r'''$SOURCE_ROOT''').expanduser().resolve())
PY
)"
MANIFEST_DIR="$(dirname "$MANIFEST")"
BUILD_DIR="$BUILD_ROOT/$DATASET_VERSION"

mkdir -p "$MANIFEST_DIR" "$NPZ_CLIPS_ROOT" "$BUILD_ROOT"

if [[ "$FORCE_CLEAN" == "1" ]]; then
  echo "[CLEAN] removing $MANIFEST"
  rm -f "$MANIFEST"
  echo "[CLEAN] removing $BUILD_DIR"
  rm -rf "$BUILD_DIR"
fi

echo "[INFO] project root:   $PROJECT_ROOT"
echo "[INFO] source root:    $SOURCE_ROOT"
echo "[INFO] manifest:       $MANIFEST"
echo "[INFO] dataset:        $DATASET_VERSION"
echo "[INFO] target fps:     $TARGET_FPS"
echo "[INFO] npz clips root: $NPZ_CLIPS_ROOT"
echo

found_sources=()
for source in "${SOURCES[@]}"; do
  input_dir="$SOURCE_ROOT/$source"
  if [[ -d "$input_dir" ]]; then
    found_sources+=("$source")
  else
    echo "[SKIP] missing source: $input_dir"
  fi
done

if [[ ${#found_sources[@]} -eq 0 ]]; then
  echo "ERROR: no twist2_full sources found under $SOURCE_ROOT" >&2
  exit 1
fi

for source in "${found_sources[@]}"; do
  input_dir="$SOURCE_ROOT/$source"
  echo "[INGEST] $source"
  python scripts/ingest_motion.py \
    --input "$input_dir" \
    --source "$source" \
    --manifest "$MANIFEST" \
    --npz_root "$NPZ_ROOT" \
    --npz_clips_root "$NPZ_CLIPS_ROOT" \
    --quality_tag legacy \
    --allow_update
  echo
done

echo "[VALIDATE] manifest"
python scripts/data/validate_dataset.py \
  --manifest "$MANIFEST" \
  --npz_root "$NPZ_ROOT"
echo

echo "[BUILD] merged train/val"
python scripts/data/build_dataset.py \
  --manifest "$MANIFEST" \
  --dataset_version "$DATASET_VERSION" \
  --npz_root "$NPZ_ROOT" \
  --build_root "$BUILD_ROOT" \
  --val_percent "$VAL_PERCENT" \
  --hash_salt "$HASH_SALT" \
  --target_fps "$TARGET_FPS"
echo

echo "[DONE]"
echo "  Manifest:    $MANIFEST"
echo "  Build dir:   $BUILD_DIR"
echo "  Train NPZ:   $BUILD_DIR/merged_train.npz"
echo "  Val NPZ:     $BUILD_DIR/merged_val.npz"
echo "  Resolved CSV:$BUILD_DIR/manifest_resolved.csv"
echo "  Build info:  $BUILD_DIR/build_info.json"
