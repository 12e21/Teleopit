"""Shared utilities for dataset review: data model, I/O, and statistics."""

from __future__ import annotations

import csv
import os
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


REVIEW_COLUMNS = [
    "clip_id",
    "source",
    "file_rel",
    "resolved_split",
    "num_frames",
    "fps",
    "duration_s",
    "weight",
    "decision",
    "difficulty",
    "issue_tags",
    "note",
    "reviewed_at",
]

VALID_DECISIONS = {"keep", "drop", "skip", ""}
VALID_DIFFICULTIES = {"easy", "medium", "hard", "bad_data", ""}


@dataclass
class ReviewRow:
    clip_id: str
    source: str
    file_rel: str
    resolved_split: str
    num_frames: int
    fps: int
    duration_s: float
    weight: float = 1.0
    decision: str = ""
    difficulty: str = ""
    issue_tags: str = ""
    note: str = ""
    reviewed_at: str = ""


@dataclass(frozen=True)
class ReviewStats:
    total: int
    reviewed: int
    keep_count: int
    drop_count: int
    skip_count: int
    progress_pct: float
    kept_duration_s: float
    kept_train_duration_s: float
    kept_val_duration_s: float
    kept_duration_by_source: dict[str, float]


def load_review_state(path: Path) -> list[ReviewRow]:
    """Load review_state.csv and return a list of ReviewRow."""
    if not path.is_file():
        raise FileNotFoundError(f"review state not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"review state is empty: {path}")
        missing = [c for c in REVIEW_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"review state missing columns: {missing}")

        rows: list[ReviewRow] = []
        for idx, raw in enumerate(reader, start=2):
            try:
                row = ReviewRow(
                    clip_id=raw["clip_id"].strip(),
                    source=raw["source"].strip(),
                    file_rel=raw["file_rel"].strip(),
                    resolved_split=raw["resolved_split"].strip(),
                    num_frames=int(raw["num_frames"]),
                    fps=int(raw["fps"]),
                    duration_s=float(raw["duration_s"]),
                    weight=float(raw["weight"]),
                    decision=raw["decision"].strip(),
                    difficulty=raw["difficulty"].strip(),
                    issue_tags=raw["issue_tags"].strip(),
                    note=raw["note"].strip(),
                    reviewed_at=raw["reviewed_at"].strip(),
                )
            except Exception as exc:
                raise ValueError(f"line {idx}: {exc}") from exc

            if row.decision not in VALID_DECISIONS:
                raise ValueError(f"line {idx}: invalid decision '{row.decision}'")
            if row.difficulty not in VALID_DIFFICULTIES:
                raise ValueError(f"line {idx}: invalid difficulty '{row.difficulty}'")
            rows.append(row)
    return rows


def save_review_state(rows: list[ReviewRow], path: Path) -> None:
    """Atomically write review_state.csv (write to .tmp, then os.replace)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".csv.tmp")
    with tmp_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(REVIEW_COLUMNS)
        for row in rows:
            writer.writerow([
                row.clip_id,
                row.source,
                row.file_rel,
                row.resolved_split,
                row.num_frames,
                row.fps,
                f"{row.duration_s:.4f}",
                row.weight,
                row.decision,
                row.difficulty,
                row.issue_tags,
                row.note,
                row.reviewed_at,
            ])
    os.replace(tmp_path, path)


def compute_review_stats(rows: list[ReviewRow]) -> ReviewStats:
    """Compute aggregate statistics from review rows."""
    total = len(rows)
    reviewed = sum(1 for r in rows if r.decision != "")
    keep_count = sum(1 for r in rows if r.decision == "keep")
    drop_count = sum(1 for r in rows if r.decision == "drop")
    skip_count = sum(1 for r in rows if r.decision == "skip")

    kept_duration_s = 0.0
    kept_train_duration_s = 0.0
    kept_val_duration_s = 0.0
    kept_duration_by_source: dict[str, float] = {}

    for r in rows:
        if r.decision != "keep":
            continue
        kept_duration_s += r.duration_s
        if r.resolved_split == "train":
            kept_train_duration_s += r.duration_s
        elif r.resolved_split == "val":
            kept_val_duration_s += r.duration_s
        kept_duration_by_source[r.source] = (
            kept_duration_by_source.get(r.source, 0.0) + r.duration_s
        )

    progress_pct = (reviewed / total * 100.0) if total > 0 else 0.0

    return ReviewStats(
        total=total,
        reviewed=reviewed,
        keep_count=keep_count,
        drop_count=drop_count,
        skip_count=skip_count,
        progress_pct=progress_pct,
        kept_duration_s=kept_duration_s,
        kept_train_duration_s=kept_train_duration_s,
        kept_val_duration_s=kept_val_duration_s,
        kept_duration_by_source=kept_duration_by_source,
    )


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()
