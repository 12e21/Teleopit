#!/usr/bin/env bash
# render_all_lafan1.sh — Batch render ALL lafan1 BVH files.
#
# For each BVH in data/lafan1/, produces 3 videos:
#   outputs/lafan1/{stem}/bvh.mp4
#   outputs/lafan1/{stem}/retarget.mp4
#   outputs/lafan1/{stem}/sim2sim.mp4
#
# Usage:
#   bash scripts/render_all_lafan1.sh --policy /path/to/policy.onnx [--max_seconds 30]
#
# Runs sequentially (GPU memory constraint). Skips already-rendered files.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
DATA_DIR="$PROJECT_ROOT/data/lafan1"
OUTPUT_DIR="$PROJECT_ROOT/outputs/lafan1"

EXTRA_ARGS=("$@")

HAS_POLICY=0
for arg in "${EXTRA_ARGS[@]}"; do
    if [[ "$arg" == "--policy" || "$arg" == --policy=* ]]; then
        HAS_POLICY=1
        break
    fi
done

if [[ $HAS_POLICY -ne 1 ]]; then
    echo "ERROR: --policy is required (ONNX exported from train_mimic checkpoint)."
    echo "Usage: bash scripts/render_all_lafan1.sh --policy /path/to/policy.onnx [--max_seconds 30]"
    exit 1
fi

export MUJOCO_GL="${MUJOCO_GL:-egl}"

mapfile -t BVH_FILES < <(find "$DATA_DIR" -maxdepth 1 -name '*.bvh' -type f | sort)
TOTAL=${#BVH_FILES[@]}

if [[ $TOTAL -eq 0 ]]; then
    echo "ERROR: No .bvh files found in $DATA_DIR"
    exit 1
fi

echo "=== Batch Render: $TOTAL lafan1 BVH files ==="
echo "    Output: $OUTPUT_DIR/{stem}/{bvh,retarget,sim2sim}.mp4"
echo "    Extra args: ${EXTRA_ARGS[*]:-none}"
echo ""

DONE=0
FAIL=0
SKIP=0

for bvh in "${BVH_FILES[@]}"; do
    stem="$(basename "$bvh" .bvh)"
    idx=$((DONE + FAIL + SKIP + 1))
    dest="$OUTPUT_DIR/$stem"

    if [[ -f "$dest/bvh.mp4" && -f "$dest/retarget.mp4" && -f "$dest/sim2sim.mp4" ]]; then
        echo "[$idx/$TOTAL] SKIP $stem (already rendered)"
        SKIP=$((SKIP + 1))
        continue
    fi

    echo "[$idx/$TOTAL] Rendering $stem ..."

    if python "$SCRIPT_DIR/render_sim.py" --bvh "$bvh" "${EXTRA_ARGS[@]}" 2>&1; then
        mkdir -p "$dest"
        flat_dir="$PROJECT_ROOT/outputs"
        for vtype in bvh retarget sim2sim; do
            src="$flat_dir/${stem}_${vtype}.mp4"
            if [[ -f "$src" ]]; then
                mv "$src" "$dest/${vtype}.mp4"
            fi
        done
        DONE=$((DONE + 1))
        echo "[$idx/$TOTAL] OK $stem"
    else
        FAIL=$((FAIL + 1))
        echo "[$idx/$TOTAL] FAIL $stem (continuing...)"
    fi

    echo ""
done

echo "=== Summary ==="
echo "  Total:   $TOTAL"
echo "  Done:    $DONE"
echo "  Skipped: $SKIP"
echo "  Failed:  $FAIL"

if [[ $FAIL -gt 0 ]]; then
    exit 1
fi
