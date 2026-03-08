from __future__ import annotations

import csv
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import yaml

from train_mimic.data.dataset_lib import (
    hash_split,
    inspect_npz,
    merge_npz_files,
    sha256_file,
    utc_now_iso,
    write_json,
)
from train_mimic.data.motion_fk import compute_npz_fk_consistency

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = PROJECT_ROOT / 'train_mimic' / 'configs' / 'datasets' / 'twist2_full.yaml'
DEFAULT_CACHE_ROOT = PROJECT_ROOT / 'data' / 'datasets' / 'cache'
DEFAULT_BUILD_ROOT = PROJECT_ROOT / 'data' / 'datasets' / 'builds'
DEFAULT_FK_SAMPLE_CLIPS = 2
DEFAULT_FK_SAMPLE_FRAMES = 16


@dataclass(frozen=True)
class DatasetSourceSpec:
    name: str
    input: str


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    target_fps: int
    val_percent: int
    hash_salt: str
    sources: list[DatasetSourceSpec]


@dataclass(frozen=True)
class DatasetClipRow:
    clip_id: str
    source: str
    file_rel: str
    num_frames: int
    fps: int
    resolved_split: str
    resolved_npz_path: str


@dataclass(frozen=True)
class DatasetPaths:
    cache_dir: Path
    clips_root: Path
    build_dir: Path


@dataclass(frozen=True)
class FkCheckSummary:
    source: str
    clip: str
    pos_max: float
    quat_mean: float
    quat_p95: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f'dataset spec not found: {spec_path}')
    payload = yaml.safe_load(spec_path.read_text(encoding='utf-8'))
    if not isinstance(payload, dict):
        raise ValueError(f'dataset spec must be a mapping: {spec_path}')

    name = str(payload.get('name', '')).strip()
    if not name:
        raise ValueError(f'dataset spec missing non-empty name: {spec_path}')

    target_fps = int(payload.get('target_fps', 0))
    if target_fps <= 0:
        raise ValueError(f'dataset spec target_fps must be > 0: {spec_path}')

    val_percent = int(payload.get('val_percent', 0))
    if val_percent <= 0 or val_percent >= 100:
        raise ValueError(f'dataset spec val_percent must be in [1, 99]: {spec_path}')

    hash_salt = str(payload.get('hash_salt', ''))
    raw_sources = payload.get('sources')
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f'dataset spec must define a non-empty sources list: {spec_path}')

    sources: list[DatasetSourceSpec] = []
    seen_names: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError(f'each source entry must be a mapping: {spec_path}')
        source_name = str(raw.get('name', '')).strip()
        source_input = str(raw.get('input', '')).strip()
        if not source_name:
            raise ValueError(f'source entry missing non-empty name: {spec_path}')
        if not source_input:
            raise ValueError(f'source {source_name!r} missing non-empty input: {spec_path}')
        if source_name in seen_names:
            raise ValueError(f'duplicate source name in spec: {source_name}')
        seen_names.add(source_name)
        sources.append(DatasetSourceSpec(name=source_name, input=source_input))

    return DatasetSpec(
        name=name,
        target_fps=target_fps,
        val_percent=val_percent,
        hash_salt=hash_salt,
        sources=sources,
    )


def resolve_dataset_paths(spec: DatasetSpec) -> DatasetPaths:
    cache_dir = DEFAULT_CACHE_ROOT / spec.name
    clips_root = cache_dir / 'npz_clips'
    build_dir = DEFAULT_BUILD_ROOT / spec.name
    return DatasetPaths(cache_dir=cache_dir, clips_root=clips_root, build_dir=build_dir)


def resolve_source_input_dir(source: DatasetSourceSpec) -> Path:
    input_dir = (PROJECT_ROOT / source.input).resolve()
    if not input_dir.is_dir():
        raise FileNotFoundError(f'source input dir not found for {source.name}: {input_dir}')
    return input_dir


def build_source_convert_command(source: DatasetSourceSpec, out_dir: Path) -> list[str]:
    return [
        sys.executable,
        str(PROJECT_ROOT / 'train_mimic' / 'scripts' / 'convert_pkl_to_npz.py'),
        '--input',
        str(resolve_source_input_dir(source)),
        '--output',
        str(out_dir),
    ]


def _source_has_cached_npz(out_dir: Path) -> bool:
    return any(out_dir.rglob('*.npz'))


def convert_sources_to_npz(
    spec: DatasetSpec,
    *,
    paths: DatasetPaths,
    force: bool = False,
    jobs: int = 1,
) -> dict[str, Path]:
    if jobs <= 0:
        raise ValueError(f'jobs must be > 0, got {jobs}')
    paths.clips_root.mkdir(parents=True, exist_ok=True)

    source_out_dirs = {source.name: paths.clips_root / source.name for source in spec.sources}
    if force:
        if paths.cache_dir.exists():
            shutil.rmtree(paths.cache_dir)
        if paths.build_dir.exists():
            shutil.rmtree(paths.build_dir)
        paths.clips_root.mkdir(parents=True, exist_ok=True)

    pending: list[tuple[DatasetSourceSpec, Path]] = []
    for source in spec.sources:
        out_dir = source_out_dirs[source.name]
        if _source_has_cached_npz(out_dir):
            print(f'[CACHE] reusing source={source.name} npz cache: {out_dir}')
            continue
        pending.append((source, out_dir))

    if not pending:
        return source_out_dirs

    if jobs == 1 or len(pending) == 1:
        for source, out_dir in pending:
            out_dir.mkdir(parents=True, exist_ok=True)
            command = build_source_convert_command(source, out_dir)
            print(f'[CONVERT] source={source.name}')
            subprocess.run(command, cwd=PROJECT_ROOT, check=True)
        return source_out_dirs

    from concurrent.futures import ThreadPoolExecutor

    def _run(pair: tuple[DatasetSourceSpec, Path]) -> None:
        source, out_dir = pair
        out_dir.mkdir(parents=True, exist_ok=True)
        command = build_source_convert_command(source, out_dir)
        print(f'[CONVERT] source={source.name}')
        subprocess.run(command, cwd=PROJECT_ROOT, check=True)

    with ThreadPoolExecutor(max_workers=min(jobs, len(pending))) as executor:
        list(executor.map(_run, pending))
    return source_out_dirs


def collect_clip_rows(spec: DatasetSpec, *, paths: DatasetPaths) -> list[DatasetClipRow]:
    rows: list[DatasetClipRow] = []
    for source in spec.sources:
        source_dir = paths.clips_root / source.name
        if not source_dir.is_dir():
            raise FileNotFoundError(f'expected converted npz dir for {source.name}: {source_dir}')
        npz_files = sorted(p for p in source_dir.rglob('*.npz') if p.name != 'merged.npz')
        if not npz_files:
            raise ValueError(f'no converted npz clips found for source {source.name}: {source_dir}')
        for npz_path in npz_files:
            meta = inspect_npz(npz_path)
            rel_under_source = npz_path.relative_to(source_dir).with_suffix('').as_posix()
            clip_id = f'{source.name}:{rel_under_source}'
            file_rel = npz_path.relative_to(PROJECT_ROOT).as_posix()
            rows.append(
                DatasetClipRow(
                    clip_id=clip_id,
                    source=source.name,
                    file_rel=file_rel,
                    num_frames=meta.num_frames,
                    fps=meta.fps,
                    resolved_split='',
                    resolved_npz_path=str(npz_path),
                )
            )
    return assign_splits(rows, spec.val_percent, spec.hash_salt)


def assign_splits(rows: list[DatasetClipRow], val_percent: int, hash_salt: str) -> list[DatasetClipRow]:
    if not rows:
        raise ValueError('no clip rows to split')
    resolved = [
        DatasetClipRow(
            clip_id=row.clip_id,
            source=row.source,
            file_rel=row.file_rel,
            num_frames=row.num_frames,
            fps=row.fps,
            resolved_split=hash_split(row.clip_id, val_percent, hash_salt),
            resolved_npz_path=row.resolved_npz_path,
        )
        for row in rows
    ]
    train_count = sum(1 for row in resolved if row.resolved_split == 'train')
    val_count = sum(1 for row in resolved if row.resolved_split == 'val')
    if train_count > 0 and val_count > 0:
        return resolved
    if len(resolved) < 2:
        raise ValueError('dataset must contain at least 2 clips to create both train and val splits')

    ordered = sorted(resolved, key=lambda row: row.clip_id)
    adjusted: list[DatasetClipRow] = []
    for idx, row in enumerate(ordered):
        split = row.resolved_split
        if val_count == 0 and idx == 0:
            split = 'val'
        elif train_count == 0 and idx == 0:
            split = 'train'
        adjusted.append(
            DatasetClipRow(
                clip_id=row.clip_id,
                source=row.source,
                file_rel=row.file_rel,
                num_frames=row.num_frames,
                fps=row.fps,
                resolved_split=split,
                resolved_npz_path=row.resolved_npz_path,
            )
        )
    return adjusted


def run_sample_fk_checks(
    rows: list[DatasetClipRow],
    *,
    sample_clips_per_source: int = DEFAULT_FK_SAMPLE_CLIPS,
    sample_frames: int = DEFAULT_FK_SAMPLE_FRAMES,
) -> list[FkCheckSummary]:
    summaries: list[FkCheckSummary] = []
    by_source: dict[str, list[DatasetClipRow]] = {}
    for row in rows:
        by_source.setdefault(row.source, []).append(row)

    for source, source_rows in sorted(by_source.items()):
        for row in sorted(source_rows, key=lambda item: item.clip_id)[:sample_clips_per_source]:
            stats = compute_npz_fk_consistency(row.resolved_npz_path, sample_count=sample_frames)
            if stats.pos_max > 1e-3 or stats.quat_mean > 0.05 or stats.quat_p95 > 0.10:
                raise ValueError(
                    f'FK consistency check failed for {row.clip_id}: '
                    f'pos_max={stats.pos_max:.6e}, quat_mean={stats.quat_mean:.6e}, '
                    f'quat_p95={stats.quat_p95:.6e}'
                )
            summaries.append(
                FkCheckSummary(
                    source=source,
                    clip=row.clip_id,
                    pos_max=stats.pos_max,
                    quat_mean=stats.quat_mean,
                    quat_p95=stats.quat_p95,
                )
            )
    return summaries


def write_manifest_resolved(rows: list[DatasetClipRow], build_dir: Path) -> Path:
    out_path = build_dir / 'manifest_resolved.csv'
    build_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open('w', encoding='utf-8', newline='') as handle:
        writer = csv.writer(handle)
        writer.writerow([
            'clip_id',
            'source',
            'file_rel',
            'num_frames',
            'fps',
            'resolved_split',
            'resolved_npz_path',
        ])
        for row in sorted(rows, key=lambda item: item.clip_id):
            writer.writerow([
                row.clip_id,
                row.source,
                row.file_rel,
                row.num_frames,
                row.fps,
                row.resolved_split,
                row.resolved_npz_path,
            ])
    return out_path


def _rows_for_split(rows: list[DatasetClipRow], split: str) -> list[Path]:
    selected = [Path(row.resolved_npz_path) for row in rows if row.resolved_split == split]
    if not selected:
        raise ValueError(f'no clips for split={split}')
    return selected


def build_dataset_from_spec(
    spec: DatasetSpec,
    *,
    force: bool = False,
    skip_fk_check: bool = False,
    skip_validate: bool = False,
    jobs: int = 1,
) -> dict[str, Any]:
    paths = resolve_dataset_paths(spec)
    convert_sources_to_npz(spec, paths=paths, force=force, jobs=jobs)
    rows = collect_clip_rows(spec, paths=paths)

    fk_checks: list[FkCheckSummary] = []
    if not skip_fk_check:
        fk_checks = run_sample_fk_checks(rows)

    train_files = _rows_for_split(rows, 'train')
    val_files = _rows_for_split(rows, 'val')
    paths.build_dir.mkdir(parents=True, exist_ok=True)
    train_out = paths.build_dir / 'train.npz'
    val_out = paths.build_dir / 'val.npz'

    train_stats = merge_npz_files(train_files, train_out, target_fps=spec.target_fps)
    val_stats = merge_npz_files(val_files, val_out, target_fps=spec.target_fps)

    manifest_path = write_manifest_resolved(rows, paths.build_dir)
    report: dict[str, Any] = {
        'dataset': spec.name,
        'built_at_utc': utc_now_iso(),
        'target_fps': spec.target_fps,
        'val_percent': spec.val_percent,
        'hash_salt': spec.hash_salt,
        'cache_dir': str(paths.cache_dir),
        'build_dir': str(paths.build_dir),
        'manifest_resolved': str(manifest_path),
        'skip_validate': bool(skip_validate),
        'skip_fk_check': bool(skip_fk_check),
        'jobs': int(jobs),
        'sources': [asdict(source) for source in spec.sources],
        'splits': {
            'train': train_stats,
            'val': val_stats,
        },
        'clip_counts': {
            'total': len(rows),
            'train': len(train_files),
            'val': len(val_files),
        },
        'fk_checks': [item.to_dict() for item in fk_checks],
    }
    write_json(paths.build_dir / 'build_info.json', report)
    return report
