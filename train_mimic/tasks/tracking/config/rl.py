"""PPO runner configuration for the official tracking task."""

from mjlab.rl import RslRlModelCfg, RslRlOnPolicyRunnerCfg, RslRlPpoAlgorithmCfg

from train_mimic.tasks.tracking.config.constants import DEFAULT_EXPERIMENT_NAME


def make_tracking_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
    """Create RL runner configuration for the official G1 tracking task."""
    return RslRlOnPolicyRunnerCfg(
        actor=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
            distribution_cfg={
                "class_name": "GaussianDistribution",
                "init_std": 1.0,
                "std_type": "scalar",
            },
        ),
        critic=RslRlModelCfg(
            hidden_dims=(512, 256, 128),
            activation="elu",
            obs_normalization=True,
        ),
        algorithm=RslRlPpoAlgorithmCfg(
            value_loss_coef=1.0,
            use_clipped_value_loss=True,
            clip_param=0.2,
            entropy_coef=0.005,
            num_learning_epochs=5,
            num_mini_batches=4,
            learning_rate=1.0e-3,
            schedule="adaptive",
            gamma=0.99,
            lam=0.95,
            desired_kl=0.01,
            max_grad_norm=1.0,
        ),
        experiment_name=DEFAULT_EXPERIMENT_NAME,
        save_interval=2000,
        num_steps_per_env=24,
        max_iterations=30_000,
        logger="tensorboard",
        upload_model=False,
    )
