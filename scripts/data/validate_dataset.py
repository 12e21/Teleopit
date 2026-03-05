#!/usr/bin/env python3
"""Validate motion dataset manifest and NPZ clips."""

from __future__ import annotations

import argparse
from pathlib import Path

from train_mimic.data.dataset_lib import load_manifest, validate_entries, write_json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate motion dataset manifest and NPZ clips.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to dataset manifest CSV")
    parser.add_argument(
        "--npz_root",
        type=str,
        default=".",
        help="Root directory used to resolve manifest file_rel (default: repo root '.')",
    )
    parser.add_argument(
        "--report",
        type=str,
        default=None,
        help="Output JSON report path (default: <manifest_stem>_validation_report.json)",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    manifest_path = Path(args.manifest).expanduser().resolve()
    npz_root = Path(args.npz_root).expanduser().resolve()

    entries = load_manifest(manifest_path)
    result = validate_entries(entries, npz_root)

    report_path = (
        Path(args.report).expanduser().resolve()
        if args.report
        else manifest_path.with_name(f"{manifest_path.stem}_validation_report.json")
    )
    result["manifest"] = str(manifest_path)
    result["npz_root"] = str(npz_root)
    write_json(report_path, result)

    print(f"[INFO] Validation report: {report_path}")
    print(
        f"[INFO] rows: total={result['rows_total']} enabled={result['rows_enabled']} "
        f"disabled={result['rows_disabled']}"
    )
    if result["warnings"]:
        print(f"[WARN] warnings: {len(result['warnings'])}")
    if not result["ok"]:
        print(f"[ERROR] validation failed: {len(result['errors'])} error(s)")
        for err in result["errors"][:20]:
            print(f"  - {err}")
        if len(result["errors"]) > 20:
            print(f"  ... {len(result['errors']) - 20} more")
        return 1

    print("[INFO] validation passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
