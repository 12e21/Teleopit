"""Tests for train_mimic.scripts.train multi-GPU launcher helpers."""

from __future__ import annotations

import argparse

import pytest

from train_mimic.scripts import train
from train_mimic.tasks.tracking.config.g1.rl_cfg import make_g1_tracking_ppo_runner_cfg
from train_mimic.tasks.tracking.config.g1_v1.rl_cfg import make_g1_tracking_ppo_runner_cfg_v1
from train_mimic.tasks.tracking.config.g1_v2.rl_cfg import make_g1_tracking_ppo_runner_cfg_v2


class _CudaStub:
    @staticmethod
    def is_available() -> bool:
        return True


class _TorchStub:
    cuda = _CudaStub()


def _args(**overrides: object) -> argparse.Namespace:
    defaults = {
        "task": "Tracking-Flat-G1-v0",
        "num_envs": 1024,
        "max_iterations": 10,
        "seed": 42,
        "wandb_project": None,
        "experiment_name": None,
        "motion_file": "data/datasets/builds/twist2_full/train.npz",
        "resume": None,
        "device": None,
        "gpu_ids": None,
        "master_port": 29500,
        "video": False,
        "video_interval": 2000,
        "video_length": 200,
    }
    defaults.update(overrides)
    return argparse.Namespace(**defaults)


class TestTrainLauncherHelpers:
    def test_parse_args_with_gpu_ids(self) -> None:
        args = train.parse_args(["--gpu_ids", "0", "2", "3", "--master_port", "29600"])
        assert args.gpu_ids == [0, 2, 3]
        assert args.master_port == 29600

    def test_should_launch_multi_gpu(self) -> None:
        args = _args(gpu_ids=[0, 1, 2, 3])
        assert train._should_launch_multi_gpu(args, env={"WORLD_SIZE": "1"}) is True
        assert train._should_launch_multi_gpu(args, env={"WORLD_SIZE": "4"}) is False

    def test_validate_multi_gpu_args_rejects_duplicates(self) -> None:
        with pytest.raises(ValueError, match="duplicates"):
            train._validate_multi_gpu_args(_args(gpu_ids=[0, 1, 1]))

    def test_filtered_argv_for_worker_removes_launcher_only_flags(self) -> None:
        argv = [
            "train.py",
            "--task",
            "Tracking-Flat-G1-v0",
            "--gpu_ids",
            "0",
            "1",
            "2",
            "3",
            "--master_port",
            "29600",
            "--num_envs",
            "1024",
        ]
        assert train._filtered_argv_for_worker(argv) == [
            "train.py",
            "--task",
            "Tracking-Flat-G1-v0",
            "--num_envs",
            "1024",
        ]

    def test_build_torchrun_command_uses_visible_gpu_count(self) -> None:
        args = _args(gpu_ids=[0, 2, 5], master_port=29600)
        argv = ["train.py", "--gpu_ids", "0", "2", "5", "--num_envs", "1024"]
        command = train._build_torchrun_command(args, argv)
        assert command[:4] == [
            "torchrun",
            "--standalone",
            "--nproc_per_node=3",
            "--master_port=29600",
        ]
        assert command[4:] == ["train.py", "--num_envs", "1024"]

    def test_build_launcher_env_sets_cuda_visible_devices(self) -> None:
        env = train._build_launcher_env(_args(gpu_ids=[3, 1]), env={"PATH": "/bin"})
        assert env["CUDA_VISIBLE_DEVICES"] == "3,1"
        assert env["PATH"] == "/bin"

    def test_resolve_device_defaults_to_local_rank_in_distributed_mode(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "2")
        assert train._resolve_device(_args(), _TorchStub()) == "cuda:2"

    def test_resolve_device_rejects_wrong_distributed_device(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("WORLD_SIZE", "4")
        monkeypatch.setenv("LOCAL_RANK", "1")
        with pytest.raises(ValueError, match="LOCAL_RANK=1"):
            train._resolve_device(_args(device="cuda:0"), _TorchStub())

    def test_main_uses_launcher_branch(self, monkeypatch: pytest.MonkeyPatch) -> None:
        called: dict[str, object] = {}

        def fake_launch(args: argparse.Namespace, argv: list[str]) -> None:
            called["gpu_ids"] = args.gpu_ids
            called["argv"] = argv

        def fake_worker(args: argparse.Namespace) -> None:
            raise AssertionError("worker should not run in launcher branch")

        monkeypatch.setattr(train, "_launch_multi_gpu", fake_launch)
        monkeypatch.setattr(train, "_run_worker", fake_worker)

        train.main(["train.py", "--gpu_ids", "0", "1", "--num_envs", "1024"])

        assert called == {
            "gpu_ids": [0, 1],
            "argv": ["train.py", "--gpu_ids", "0", "1", "--num_envs", "1024"],
        }


def test_tracking_runner_configs_disable_model_upload() -> None:
    assert make_g1_tracking_ppo_runner_cfg().upload_model is False
    assert make_g1_tracking_ppo_runner_cfg_v1().upload_model is False
    assert make_g1_tracking_ppo_runner_cfg_v2().upload_model is False
