#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json

from train_mimic.data.dataset_v2 import DEFAULT_SPEC_PATH, build_dataset_from_spec, load_dataset_spec


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='Build dataset from lightweight YAML spec.')
    parser.add_argument('--spec', type=str, default=str(DEFAULT_SPEC_PATH), help='YAML dataset spec path')
    parser.add_argument('--force', action='store_true', help='Delete dataset cache/build outputs before rebuild')
    parser.add_argument('--skip_fk_check', action='store_true', help='Skip sampled FK consistency checks')
    parser.add_argument('--skip_validate', action='store_true', help='Reserved lightweight flag; kept for CLI stability')
    parser.add_argument('--jobs', type=int, default=1, help='Number of source-level conversion jobs (default: 1)')
    parser.add_argument('--json', action='store_true', help='Print final build_info JSON to stdout')
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    spec = load_dataset_spec(args.spec)
    report = build_dataset_from_spec(
        spec,
        force=args.force,
        skip_fk_check=args.skip_fk_check,
        skip_validate=args.skip_validate,
        jobs=args.jobs,
    )
    print(f"[DONE] dataset={report['dataset']}")
    print(f"[DONE] train={report['splits']['train']['output']}")
    print(f"[DONE] val={report['splits']['val']['output']}")
    print(f"[DONE] build_info={report['build_dir']}/build_info.json")
    if args.json:
        print(json.dumps(report, ensure_ascii=False, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
