#!/usr/bin/env bash
# render_val_examples.sh — Render a few val-set clips with benchmark.py --video.
#
# Keeps things intentionally simple:
# - reads val clips from manifest_resolved.csv
# - picks the first N matching clips
# - runs benchmark.py once per clip
# - does NOT set MUJOCO_GL; set it yourself before running if needed
#
# Usage:
#   bash scripts/render_val_examples.sh --checkpoint ckpt/model_10000.pt
#   bash scripts/render_val_examples.sh --checkpoint ckpt/model_10000.pt --num 5
#   bash scripts/render_val_examples.sh --checkpoint ckpt/model_10000.pt --clip_contains walk

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

CHECKPOINT=""
DATASET_VERSION="twist2_full_v1_30hz"
NUM=3
CLIP_CONTAINS=""
TASK="Tracking-Flat-G1-v0"

usage() {
    cat <<USAGE
Usage:
  bash scripts/render_val_examples.sh --checkpoint <model.pt> [options]

Options:
  --checkpoint <path>       Required. Path to checkpoint (.pt)
  --dataset_version <name>  Build version under data/motion/builds/ (default: ${DATASET_VERSION})
  --num <n>                 Number of val clips to render (default: ${NUM})
  --clip_contains <text>    Only render clips whose path contains this substring
  --task <name>             Gym task name (default: ${TASK})
  --help                    Show this help
USAGE
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --checkpoint)
            CHECKPOINT="$2"
            shift 2
            ;;
        --dataset_version)
            DATASET_VERSION="$2"
            shift 2
            ;;
        --num)
            NUM="$2"
            shift 2
            ;;
        --clip_contains)
            CLIP_CONTAINS="$2"
            shift 2
            ;;
        --task)
            TASK="$2"
            shift 2
            ;;
        --help|-h)
            usage
            exit 0
            ;;
        *)
            echo "ERROR: unknown argument: $1"
            echo
            usage
            exit 1
            ;;
    esac
done

if [[ -z "$CHECKPOINT" ]]; then
    echo "ERROR: --checkpoint is required"
    exit 1
fi

if [[ ! -f "$CHECKPOINT" ]]; then
    echo "ERROR: checkpoint not found: $CHECKPOINT"
    exit 1
fi

if ! [[ "$NUM" =~ ^[0-9]+$ ]] || [[ "$NUM" -le 0 ]]; then
    echo "ERROR: --num must be a positive integer"
    exit 1
fi

MANIFEST_PATH="$PROJECT_ROOT/data/motion/builds/$DATASET_VERSION/manifest_resolved.csv"
if [[ ! -f "$MANIFEST_PATH" ]]; then
    echo "ERROR: manifest_resolved.csv not found: $MANIFEST_PATH"
    exit 1
fi

mapfile -t SELECTED < <(
python - "$MANIFEST_PATH" "$NUM" "$CLIP_CONTAINS" <<'PY'
import csv
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
num = int(sys.argv[2])
clip_contains = sys.argv[3].strip().lower()

selected = []
with manifest.open() as f:
    reader = csv.DictReader(f)
    split_key = 'resolved_split' if 'resolved_split' in reader.fieldnames else 'split'
    for row in reader:
        if (row.get(split_key, '') or '').strip().lower() != 'val':
            continue
        file_rel = (row.get('file_rel', '') or '').strip()
        clip_id = (row.get('clip_id', '') or '').strip()
        if clip_contains and clip_contains not in file_rel.lower() and clip_contains not in clip_id.lower():
            continue
        frames = int((row.get('num_frames', '0') or '0').strip())
        selected.append((clip_id, file_rel, frames))
        if len(selected) >= num:
            break

for clip_id, file_rel, frames in selected:
    print(f"{clip_id}\t{file_rel}\t{frames}")
PY
)

if [[ ${#SELECTED[@]} -eq 0 ]]; then
    echo "ERROR: no val clips matched"
    exit 1
fi

OUT_ROOT="$PROJECT_ROOT/benchmark_results/videos/val_examples/$(basename "$CHECKPOINT" .pt)"
mkdir -p "$OUT_ROOT"

echo "checkpoint: $CHECKPOINT"
echo "manifest:   $MANIFEST_PATH"
echo "output:     $OUT_ROOT"
echo "clips:      ${#SELECTED[@]}"
echo ""

INDEX=0
for row in "${SELECTED[@]}"; do
    clip_id="${row%%$'\t'*}"
    rest="${row#*$'\t'}"
    file_rel="${rest%%$'\t'*}"
    frames="${rest##*$'\t'}"
    motion_file="$PROJECT_ROOT/$file_rel"
    stem="$(basename "$motion_file" .npz)"
    out_dir="$OUT_ROOT/$stem"
    mkdir -p "$out_dir"

    eval_steps="$frames"
    video_length="$frames"
    if [[ "$video_length" -gt 600 ]]; then
        video_length=600
    fi

    echo "[$((INDEX + 1))/${#SELECTED[@]}] $stem"
    python "$PROJECT_ROOT/train_mimic/scripts/benchmark.py" \
        --task "$TASK" \
        --checkpoint "$CHECKPOINT" \
        --motion_file "$motion_file" \
        --num_envs 1 \
        --num_eval_steps "$eval_steps" \
        --warmup_steps 30 \
        --video \
        --video_length "$video_length" \
        --video_folder "$out_dir"
    echo ""

    INDEX=$((INDEX + 1))
done
