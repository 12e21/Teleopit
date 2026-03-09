"""G1 tracking task registration (v1 – general motion)."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .flat_env_cfg import make_g1_flat_tracking_env_cfg_v1
from .rl_cfg import make_g1_tracking_ppo_runner_cfg_v1

register_mjlab_task(
    task_id="Tracking-Flat-G1-v1",
    env_cfg=make_g1_flat_tracking_env_cfg_v1(),
    play_env_cfg=make_g1_flat_tracking_env_cfg_v1(play=True),
    rl_cfg=make_g1_tracking_ppo_runner_cfg_v1(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
