"""G1 tracking task registration (v2)."""

from mjlab.tasks.registry import register_mjlab_task

from train_mimic.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .flat_env_cfg import make_g1_flat_tracking_env_cfg_v2
from .rl_cfg import make_g1_tracking_ppo_runner_cfg_v2

register_mjlab_task(
    task_id="Tracking-Flat-G1-v2",
    env_cfg=make_g1_flat_tracking_env_cfg_v2(),
    play_env_cfg=make_g1_flat_tracking_env_cfg_v2(play=True),
    rl_cfg=make_g1_tracking_ppo_runner_cfg_v2(),
    runner_cls=MotionTrackingOnPolicyRunner,
)

# 154D variant: no base_pos / base_lin_vel (for real-robot deployment)
register_mjlab_task(
    task_id="Tracking-Flat-G1-v2-NoStateEst",
    env_cfg=make_g1_flat_tracking_env_cfg_v2(has_state_estimation=False),
    play_env_cfg=make_g1_flat_tracking_env_cfg_v2(has_state_estimation=False, play=True),
    rl_cfg=make_g1_tracking_ppo_runner_cfg_v2(),
    runner_cls=MotionTrackingOnPolicyRunner,
)
