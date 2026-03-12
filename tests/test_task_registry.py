"""Tests for mjlab task registration used by train_mimic."""

from __future__ import annotations

from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner


def test_no_state_estimation_tasks_are_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls

    expected_tasks = (
        "Tracking-Flat-G1-v0-NoStateEst",
        "Tracking-Flat-G1-v1-NoStateEst",
    )

    for task_name in expected_tasks:
        env_cfg = load_env_cfg(task_name)
        actor_terms = env_cfg.observations["actor"].terms

        assert "motion_anchor_pos_b" not in actor_terms
        assert "base_lin_vel" not in actor_terms
        assert "motion_anchor_ori_b" in actor_terms
        assert "base_ang_vel" in actor_terms

        assert load_rl_cfg(task_name) is not None
        assert load_runner_cls(task_name) is MotionTrackingOnPolicyRunner


def test_no_state_estimation_play_env_is_registered() -> None:
    import mjlab.tasks  # noqa: F401
    import train_mimic.tasks  # noqa: F401
    from mjlab.tasks.registry import load_env_cfg

    play_cfg = load_env_cfg("Tracking-Flat-G1-v0-NoStateEst", play=True)
    actor_terms = play_cfg.observations["actor"].terms

    assert "motion_anchor_pos_b" not in actor_terms
    assert "base_lin_vel" not in actor_terms
    assert play_cfg.observations["actor"].enable_corruption is False
