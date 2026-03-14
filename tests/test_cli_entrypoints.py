from __future__ import annotations

import subprocess
import sys
from pathlib import Path


def _run_cli(project_root: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "scripts/run_sim.py", *args],
        cwd=project_root,
        capture_output=True,
        text=True,
        check=False,
    )


def test_run_sim_help_uses_hydra_cli(project_root: Path) -> None:
    result = _run_cli(project_root, "--help")

    assert result.returncode == 0, result.stderr
    assert "Powered by Hydra" in result.stdout


def test_run_sim_hydra_help_lists_hydra_flags(project_root: Path) -> None:
    result = _run_cli(project_root, "--hydra-help")

    assert result.returncode == 0, result.stderr
    assert "--cfg" in result.stdout
    assert "--multirun" in result.stdout


def test_run_sim_cfg_job_uses_hydra_cli(project_root: Path) -> None:
    result = _run_cli(project_root, "--cfg", "job")

    assert result.returncode == 0, result.stderr
    assert "controller:" in result.stdout
    assert "input:" in result.stdout
