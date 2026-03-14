from __future__ import annotations

import csv
import os
import shutil
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

import mujoco
import yaml

from train_mimic.data.dataset_lib import (
    hash_split,
    inspect_npz,
    merge_npz_files,
    utc_now_iso,
    write_json,
)
from train_mimic.data.motion_fk import MotionFkExtractor, compute_npz_fk_consistency
from train_mimic.scripts.convert_pkl_to_npz import convert_pkl_to_npz
from teleopit.retargeting.export_pkl import convert_bvh_to_retarget_pkl, mocap_xml_path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_SPEC_PATH = PROJECT_ROOT / "train_mimic" / "configs" / "datasets" / "twist2_full.yaml"
DEFAULT_DATASETS_ROOT = PROJECT_ROOT / "data" / "datasets"
DEFAULT_FK_SAMPLE_CLIPS = 2
DEFAULT_FK_SAMPLE_FRAMES = 16
SOURCE_TYPES = {"bvh", "pkl", "npz"}
BVH_FORMATS = {"lafan1", "hc_mocap", "nokov"}
SUPPORTED_BVH_ROBOT_NAME = "unitree_g1"

_SOURCE_SUFFIXES = {
    "bvh": ".bvh",
    "pkl": ".pkl",
    "npz": ".npz",
}
_DATASET_ROOT_MARKERS = {
    "train.npz",
    "val.npz",
    "manifest_resolved.csv",
    "build_info.json",
}

_PROCESS_FK_EXTRACTOR: MotionFkExtractor | None = None
_PROCESS_BVH_MODEL_CACHE: dict[str, mujoco.MjModel] = {}


def default_jobs() -> int:
    cpu_count = os.cpu_count() or 1
    return max(1, min(cpu_count, 8))


DEFAULT_JOBS = default_jobs()


@dataclass(frozen=True)
class DatasetSourceSpec:
    name: str
    type: str
    input: str
    weight: float = 1.0
    bvh_format: str | None = None
    robot_name: str = "unitree_g1"
    max_frames: int = 0


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
    weight: float = 1.0


@dataclass(frozen=True)
class DatasetPaths:
    dataset_dir: Path
    clips_root: Path


@dataclass(frozen=True)
class FkCheckSummary:
    source: str
    clip: str
    pos_max: float
    quat_mean: float
    quat_p95: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class SourceInputFile:
    path: Path
    rel_no_suffix: Path


@dataclass(frozen=True)
class ConversionTask:
    source_name: str
    source_type: str
    input_path: str
    output_path: str
    bvh_format: str | None = None
    robot_name: str = "unitree_g1"
    max_frames: int = 0
    mocap_xml: str | None = None


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _validate_source_type(raw_type: object, spec_path: Path, source_name: str) -> str:
    source_type = str(raw_type or "").strip()
    if source_type not in SOURCE_TYPES:
        raise ValueError(
            f"source {source_name!r} has invalid type {source_type!r} in {spec_path}. "
            f"Expected one of {sorted(SOURCE_TYPES)}."
        )
    return source_type


def load_dataset_spec(path: str | Path) -> DatasetSpec:
    spec_path = Path(path).expanduser().resolve()
    if not spec_path.is_file():
        raise FileNotFoundError(f"dataset spec not found: {spec_path}")
    payload = yaml.safe_load(spec_path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"dataset spec must be a mapping: {spec_path}")

    name = str(payload.get("name", "")).strip()
    if not name:
        raise ValueError(f"dataset spec missing non-empty name: {spec_path}")

    target_fps = int(payload.get("target_fps", 0))
    if target_fps <= 0:
        raise ValueError(f"dataset spec target_fps must be > 0: {spec_path}")

    val_percent = int(payload.get("val_percent", 0))
    if val_percent <= 0 or val_percent >= 100:
        raise ValueError(f"dataset spec val_percent must be in [1, 99]: {spec_path}")

    hash_salt = str(payload.get("hash_salt", ""))
    raw_sources = payload.get("sources")
    if not isinstance(raw_sources, list) or not raw_sources:
        raise ValueError(f"dataset spec must define a non-empty sources list: {spec_path}")

    sources: list[DatasetSourceSpec] = []
    seen_names: set[str] = set()
    for raw in raw_sources:
        if not isinstance(raw, dict):
            raise ValueError(f"each source entry must be a mapping: {spec_path}")

        source_name = str(raw.get("name", "")).strip()
        if not source_name:
            raise ValueError(f"source entry missing non-empty name: {spec_path}")
        if source_name in seen_names:
            raise ValueError(f"duplicate source name in spec: {source_name}")
        seen_names.add(source_name)

        source_type = _validate_source_type(raw.get("type"), spec_path, source_name)
        source_input = str(raw.get("input", "")).strip()
        if not source_input:
            raise ValueError(f"source {source_name!r} missing non-empty input: {spec_path}")

        source_weight = float(raw.get("weight", 1.0))
        if source_weight <= 0:
            raise ValueError(f"source {source_name!r} has non-positive weight: {source_weight}")

        bvh_format = raw.get("bvh_format")
        if source_type == "bvh":
            bvh_format = str(bvh_format or "").strip()
            if bvh_format not in BVH_FORMATS:
                raise ValueError(
                    f"source {source_name!r} requires bvh_format in {sorted(BVH_FORMATS)}: {spec_path}"
                )
        else:
            bvh_format = None

        robot_name = str(raw.get("robot_name", "unitree_g1")).strip() or "unitree_g1"
        max_frames = int(raw.get("max_frames", 0))
        if max_frames < 0:
            raise ValueError(f"source {source_name!r} has negative max_frames: {max_frames}")

        sources.append(
            DatasetSourceSpec(
                name=source_name,
                type=source_type,
                input=source_input,
                weight=source_weight,
                bvh_format=bvh_format,
                robot_name=robot_name,
                max_frames=max_frames,
            )
        )

    return DatasetSpec(
        name=name,
        target_fps=target_fps,
        val_percent=val_percent,
        hash_salt=hash_salt,
        sources=sources,
    )


def resolve_dataset_paths(spec: DatasetSpec, *, output_root: str | Path | None = None) -> DatasetPaths:
    base_root = DEFAULT_DATASETS_ROOT if output_root is None else Path(output_root).expanduser().resolve()
    dataset_dir = base_root / spec.name
    clips_root = dataset_dir / "clips"
    return DatasetPaths(dataset_dir=dataset_dir, clips_root=clips_root)


def resolve_source_input_path(source: DatasetSourceSpec) -> Path:
    candidate = Path(source.input).expanduser()
    input_path = candidate.resolve() if candidate.is_absolute() else (PROJECT_ROOT / candidate).resolve()
    if not input_path.exists():
        raise FileNotFoundError(f"source input not found for {source.name}: {input_path}")
    return input_path


def _ensure_not_dataset_root_npz_input(source: DatasetSourceSpec, input_path: Path) -> None:
    if source.type != "npz" or not input_path.is_dir():
        return
    has_clips_dir = (input_path / "clips").is_dir()
    has_dataset_outputs = any((input_path / name).exists() for name in _DATASET_ROOT_MARKERS)
    if has_clips_dir and has_dataset_outputs:
        raise ValueError(
            f"npz source {source.name!r} points at dataset root {input_path}, which contains merged "
            "dataset outputs. Point the source at the clip directory instead, for example "
            f"{input_path / 'clips'}, or at a specific .npz clip file."
        )


def _resolve_bvh_xml_path(source: DatasetSourceSpec) -> Path:
    assert source.type == "bvh"
    if source.robot_name != SUPPORTED_BVH_ROBOT_NAME:
        raise ValueError(
            f"source {source.name!r} uses robot_name={source.robot_name!r}, but dataset BVH conversion "
            f"currently supports only {SUPPORTED_BVH_ROBOT_NAME!r}. Use robot_name={SUPPORTED_BVH_ROBOT_NAME!r}."
        )
    xml_path = mocap_xml_path(PROJECT_ROOT, SUPPORTED_BVH_ROBOT_NAME)
    if not xml_path.is_file():
        raise FileNotFoundError(f"MuJoCo XML not found for BVH conversion: {xml_path}")
    return xml_path


def _collect_source_files(source: DatasetSourceSpec) -> tuple[list[SourceInputFile], Path]:
    input_path = resolve_source_input_path(source)
    _ensure_not_dataset_root_npz_input(source, input_path)
    suffix = _SOURCE_SUFFIXES[source.type]

    if input_path.is_file():
        if input_path.suffix.lower() != suffix:
            raise ValueError(
                f"source {source.name} expected {suffix} input, got file {input_path.name}"
            )
        return [SourceInputFile(path=input_path, rel_no_suffix=Path(input_path.stem))], input_path.parent

    if not input_path.is_dir():
        raise FileNotFoundError(f"source input is neither file nor directory: {input_path}")

    files = sorted(
        path for path in input_path.rglob("*")
        if path.is_file()
        and path.suffix.lower() == suffix
        and not (source.type == "npz" and path.name == "merged.npz")
    )
    if not files:
        raise ValueError(f"no {suffix} files found for source {source.name}: {input_path}")

    items = [
        SourceInputFile(path=path, rel_no_suffix=path.relative_to(input_path).with_suffix(""))
        for path in files
    ]
    return items, input_path


def build_source_conversion_tasks(source: DatasetSourceSpec, out_dir: Path) -> list[ConversionTask]:
    items, _scan_root = _collect_source_files(source)
    output_map: dict[Path, SourceInputFile] = {}
    xml_path = None
    if source.type == "bvh":
        xml_path = str(_resolve_bvh_xml_path(source))

    tasks: list[ConversionTask] = []
    for item in items:
        out_npz = out_dir / item.rel_no_suffix.with_suffix(".npz")
        if out_npz in output_map:
            prev = output_map[out_npz]
            raise ValueError(
                f"output collision detected for source {source.name}: "
                f"{prev.path} and {item.path} -> {out_npz}"
            )
        output_map[out_npz] = item
        tasks.append(
            ConversionTask(
                source_name=source.name,
                source_type=source.type,
                input_path=str(item.path),
                output_path=str(out_npz),
                bvh_format=source.bvh_format,
                robot_name=source.robot_name,
                max_frames=source.max_frames,
                mocap_xml=xml_path,
            )
        )
    return tasks


def _source_has_cached_npz(out_dir: Path) -> bool:
    return out_dir.is_dir() and any(path.name != "merged.npz" for path in out_dir.rglob("*.npz"))


def _pending_tasks(tasks: list[ConversionTask]) -> list[ConversionTask]:
    return [task for task in tasks if not Path(task.output_path).is_file()]


def _get_fk_extractor() -> MotionFkExtractor:
    global _PROCESS_FK_EXTRACTOR
    if _PROCESS_FK_EXTRACTOR is None:
        _PROCESS_FK_EXTRACTOR = MotionFkExtractor()
    return _PROCESS_FK_EXTRACTOR


def _get_bvh_model(xml_path: str) -> mujoco.MjModel:
    model = _PROCESS_BVH_MODEL_CACHE.get(xml_path)
    if model is None:
        model = mujoco.MjModel.from_xml_path(xml_path)
        _PROCESS_BVH_MODEL_CACHE[xml_path] = model
    return model


def _convert_task(task: ConversionTask) -> str:
    input_path = Path(task.input_path)
    output_path = Path(task.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    try:
        if task.source_type == "npz":
            if input_path.resolve() != output_path.resolve():
                shutil.copy2(input_path, output_path)
            inspect_npz(output_path)
            return str(output_path)

        extractor = _get_fk_extractor()
        if task.source_type == "pkl":
            convert_pkl_to_npz(str(input_path), str(output_path), extractor=extractor)
            inspect_npz(output_path)
            return str(output_path)

        if task.source_type == "bvh":
            if task.bvh_format is None or task.mocap_xml is None:
                raise ValueError("bvh conversion task missing bvh_format or mocap_xml")
            model = _get_bvh_model(task.mocap_xml)
            with TemporaryDirectory(prefix="teleopit_dataset_bvh_") as tmp_dir:
                tmp_pkl = Path(tmp_dir) / f"{input_path.stem}.pkl"
                convert_bvh_to_retarget_pkl(
                    input_path,
                    tmp_pkl,
                    task.bvh_format,
                    task.robot_name,
                    task.max_frames,
                    model,
                )
                convert_pkl_to_npz(str(tmp_pkl), str(output_path), extractor=extractor)
            inspect_npz(output_path)
            return str(output_path)

        raise ValueError(f"unsupported source type: {task.source_type}")
    except Exception as exc:
        raise RuntimeError(
            f"failed converting source={task.source_name} input={input_path}: {exc}"
        ) from exc


def _run_conversion_tasks_serial(tasks: list[ConversionTask]) -> None:
    total = len(tasks)
    for idx, task in enumerate(tasks, start=1):
        _convert_task(task)
        print(f"[CONVERT] {idx}/{total} source={task.source_name} -> {_display_path(Path(task.output_path))}")


def run_conversion_tasks(tasks: list[ConversionTask], *, jobs: int = DEFAULT_JOBS) -> None:
    if jobs <= 0:
        raise ValueError(f"jobs must be > 0, got {jobs}")
    if not tasks:
        return

    total = len(tasks)
    if jobs == 1 or total == 1:
        _run_conversion_tasks_serial(tasks)
        return

    max_workers = min(jobs, total)
    try:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            future_map = {executor.submit(_convert_task, task): task for task in tasks}
            completed = 0
            try:
                for future in as_completed(future_map):
                    task = future_map[future]
                    future.result()
                    completed += 1
                    print(
                        f"[CONVERT] {completed}/{total} "
                        f"source={task.source_name} -> {_display_path(Path(task.output_path))}"
                    )
            except Exception:
                for future in future_map:
                    future.cancel()
                raise
    except (PermissionError, OSError) as exc:
        print(f"[WARN] process pool unavailable ({exc}); falling back to serial conversion")
        _run_conversion_tasks_serial(tasks)


def convert_source_to_npz_clips(
    source: DatasetSourceSpec,
    output_dir: Path,
    *,
    force: bool = False,
    jobs: int = DEFAULT_JOBS,
) -> dict[str, Any]:
    if force and output_dir.exists():
        shutil.rmtree(output_dir)
    tasks = build_source_conversion_tasks(source, output_dir)
    pending = _pending_tasks(tasks)
    if tasks and not pending and _source_has_cached_npz(output_dir):
        print(f"[CACHE] reusing source={source.name} clips: {_display_path(output_dir)}")
    else:
        output_dir.mkdir(parents=True, exist_ok=True)
        run_conversion_tasks(pending or tasks, jobs=jobs)

    npz_files = sorted(path for path in output_dir.rglob("*.npz") if path.name != "merged.npz")
    if not npz_files:
        raise ValueError(f"no converted npz clips found for source {source.name}: {output_dir}")
    return {
        "source": source.name,
        "type": source.type,
        "output_dir": str(output_dir),
        "clips": len(npz_files),
    }


def convert_sources_to_npz(
    spec: DatasetSpec,
    *,
    paths: DatasetPaths,
    force: bool = False,
    jobs: int = DEFAULT_JOBS,
) -> dict[str, Path]:
    if jobs <= 0:
        raise ValueError(f"jobs must be > 0, got {jobs}")
    if force and paths.dataset_dir.exists():
        shutil.rmtree(paths.dataset_dir)
    paths.clips_root.mkdir(parents=True, exist_ok=True)

    source_out_dirs = {source.name: paths.clips_root / source.name for source in spec.sources}
    pending_tasks: list[ConversionTask] = []
    for source in spec.sources:
        out_dir = source_out_dirs[source.name]
        tasks = build_source_conversion_tasks(source, out_dir)
        pending = _pending_tasks(tasks)
        if tasks and not pending and _source_has_cached_npz(out_dir):
            print(f"[CACHE] reusing source={source.name} clips: {_display_path(out_dir)}")
            continue
        out_dir.mkdir(parents=True, exist_ok=True)
        pending_tasks.extend(pending or tasks)

    run_conversion_tasks(pending_tasks, jobs=jobs)
    return source_out_dirs


def collect_clip_rows(spec: DatasetSpec, *, paths: DatasetPaths) -> list[DatasetClipRow]:
    rows: list[DatasetClipRow] = []
    for source in spec.sources:
        source_dir = paths.clips_root / source.name
        if not source_dir.is_dir():
            raise FileNotFoundError(f"expected converted npz dir for {source.name}: {source_dir}")
        npz_files = sorted(path for path in source_dir.rglob("*.npz") if path.name != "merged.npz")
        if not npz_files:
            raise ValueError(f"no converted npz clips found for source {source.name}: {source_dir}")
        for npz_path in npz_files:
            meta = inspect_npz(npz_path)
            rel_under_source = npz_path.relative_to(source_dir).with_suffix("").as_posix()
            rows.append(
                DatasetClipRow(
                    clip_id=f"{source.name}:{rel_under_source}",
                    source=source.name,
                    file_rel=_display_path(npz_path),
                    num_frames=meta.num_frames,
                    fps=meta.fps,
                    resolved_split="",
                    resolved_npz_path=str(npz_path),
                    weight=source.weight,
                )
            )
    return assign_splits(rows, spec.val_percent, spec.hash_salt)


def assign_splits(rows: list[DatasetClipRow], val_percent: int, hash_salt: str) -> list[DatasetClipRow]:
    if not rows:
        raise ValueError("no clip rows to split")

    resolved = [
        DatasetClipRow(
            clip_id=row.clip_id,
            source=row.source,
            file_rel=row.file_rel,
            num_frames=row.num_frames,
            fps=row.fps,
            resolved_split=hash_split(row.clip_id, val_percent, hash_salt),
            resolved_npz_path=row.resolved_npz_path,
            weight=row.weight,
        )
        for row in rows
    ]
    train_count = sum(1 for row in resolved if row.resolved_split == "train")
    val_count = sum(1 for row in resolved if row.resolved_split == "val")
    if train_count > 0 and val_count > 0:
        return resolved
    if len(resolved) < 2:
        raise ValueError("dataset must contain at least 2 clips to create both train and val splits")

    ordered = sorted(resolved, key=lambda row: row.clip_id)
    adjusted: list[DatasetClipRow] = []
    for idx, row in enumerate(ordered):
        split = row.resolved_split
        if val_count == 0 and idx == 0:
            split = "val"
        elif train_count == 0 and idx == 0:
            split = "train"
        adjusted.append(
            DatasetClipRow(
                clip_id=row.clip_id,
                source=row.source,
                file_rel=row.file_rel,
                num_frames=row.num_frames,
                fps=row.fps,
                resolved_split=split,
                resolved_npz_path=row.resolved_npz_path,
                weight=row.weight,
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
                    f"FK consistency check failed for {row.clip_id}: "
                    f"pos_max={stats.pos_max:.6e}, quat_mean={stats.quat_mean:.6e}, "
                    f"quat_p95={stats.quat_p95:.6e}"
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


def write_manifest_resolved(rows: list[DatasetClipRow], dataset_dir: Path) -> Path:
    out_path = dataset_dir / "manifest_resolved.csv"
    dataset_dir.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "clip_id",
                "source",
                "file_rel",
                "num_frames",
                "fps",
                "resolved_split",
                "resolved_npz_path",
                "weight",
            ]
        )
        for row in sorted(rows, key=lambda item: item.clip_id):
            writer.writerow(
                [
                    row.clip_id,
                    row.source,
                    row.file_rel,
                    row.num_frames,
                    row.fps,
                    row.resolved_split,
                    row.resolved_npz_path,
                    row.weight,
                ]
            )
    return out_path


def _rows_for_split(rows: list[DatasetClipRow], split: str) -> tuple[list[Path], list[float]]:
    selected = [(Path(row.resolved_npz_path), row.weight) for row in rows if row.resolved_split == split]
    if not selected:
        raise ValueError(f"no clips for split={split}")
    files, weights = zip(*selected)
    return list(files), list(weights)


def build_dataset_from_spec(
    spec: DatasetSpec,
    *,
    force: bool = False,
    skip_fk_check: bool = False,
    skip_validate: bool = False,
    jobs: int = DEFAULT_JOBS,
    output_root: str | Path | None = None,
) -> dict[str, Any]:
    paths = resolve_dataset_paths(spec, output_root=output_root)
    convert_sources_to_npz(spec, paths=paths, force=force, jobs=jobs)
    rows = collect_clip_rows(spec, paths=paths)

    fk_checks: list[FkCheckSummary] = []
    if not skip_fk_check:
        fk_checks = run_sample_fk_checks(rows)

    train_files, train_weights = _rows_for_split(rows, "train")
    val_files, val_weights = _rows_for_split(rows, "val")
    paths.dataset_dir.mkdir(parents=True, exist_ok=True)
    train_out = paths.dataset_dir / "train.npz"
    val_out = paths.dataset_dir / "val.npz"

    train_stats = merge_npz_files(train_files, train_out, target_fps=spec.target_fps, weights=train_weights)
    val_stats = merge_npz_files(val_files, val_out, target_fps=spec.target_fps, weights=val_weights)

    manifest_path = write_manifest_resolved(rows, paths.dataset_dir)
    report: dict[str, Any] = {
        "dataset": spec.name,
        "built_at_utc": utc_now_iso(),
        "target_fps": spec.target_fps,
        "val_percent": spec.val_percent,
        "hash_salt": spec.hash_salt,
        "dataset_dir": str(paths.dataset_dir),
        "build_dir": str(paths.dataset_dir),
        "clips_dir": str(paths.clips_root),
        "manifest_resolved": str(manifest_path),
        "skip_validate": bool(skip_validate),
        "skip_fk_check": bool(skip_fk_check),
        "jobs": int(jobs),
        "sources": [asdict(source) for source in spec.sources],
        "splits": {
            "train": train_stats,
            "val": val_stats,
        },
        "clip_counts": {
            "total": len(rows),
            "train": len(train_files),
            "val": len(val_files),
        },
        "fk_checks": [item.to_dict() for item in fk_checks],
    }
    write_json(paths.dataset_dir / "build_info.json", report)
    return report
