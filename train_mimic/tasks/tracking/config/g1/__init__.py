"""G1 tracking task registration."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .flat_env_cfg import make_g1_flat_tracking_env_cfg
from .rl_cfg import make_g1_tracking_ppo_runner_cfg

register_mjlab_task(
    task_id="Tracking-Flat-G1-v0",
    env_cfg=make_g1_flat_tracking_env_cfg(),
    play_env_cfg=make_g1_flat_tracking_env_cfg(play=True),
    rl_cfg=make_g1_tracking_ppo_runner_cfg(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
