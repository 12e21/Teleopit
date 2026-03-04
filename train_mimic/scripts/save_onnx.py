#!/usr/bin/env python3
"""Export trained tracking policy to ONNX format for hardware deployment.

Simplified for standard MLP (no Conv1d motion encoder). Loads rsl_rl
checkpoint and exports actor network with empirical normalization baked in.

Usage:
    python train_mimic/scripts/save_onnx.py \
        --checkpoint logs/rsl_rl/g1_tracking/.../model_30000.pt \
        --output policy.onnx
"""

from __future__ import annotations

import argparse
import os

import torch
import torch.nn as nn


class PolicyExportWrapper(nn.Module):
    """Wraps rsl_rl ActorCritic actor for ONNX export with normalization."""

    def __init__(self, actor: nn.Module, normalizer: nn.Module | None = None) -> None:
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        if self.normalizer is not None:
            obs = self.normalizer(obs)
        return self.actor(obs)


def export_policy_as_onnx(
    checkpoint_path: str,
    output_path: str,
    num_observations: int = 189,
    num_actions: int = 29,
    actor_hidden_dims: list[int] | None = None,
    activation: str = "elu",
) -> None:
    """Export policy to ONNX.

    Args:
        checkpoint_path: Path to rsl_rl checkpoint (.pt)
        output_path: Output ONNX file path
        num_observations: Observation dimension (~189 for mjlab tracking)
        num_actions: Action dimension (29 for G1)
        actor_hidden_dims: MLP hidden dimensions
        activation: Activation function name
    """
    if actor_hidden_dims is None:
        actor_hidden_dims = [512, 256, 128]

    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # rsl_rl checkpoint format: {'model_state_dict': ..., ...}
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Build actor MLP matching rsl_rl ActorCritic structure
    activation_fn = {"elu": nn.ELU, "relu": nn.ReLU, "tanh": nn.Tanh}[activation]
    layers: list[nn.Module] = []
    in_dim = num_observations
    for hidden_dim in actor_hidden_dims:
        layers.append(nn.Linear(in_dim, hidden_dim))
        layers.append(activation_fn())
        in_dim = hidden_dim
    layers.append(nn.Linear(in_dim, num_actions))
    actor = nn.Sequential(*layers)

    # Load actor weights (rsl_rl keys: actor.0.weight, actor.0.bias, ...)
    actor_state = {}
    for key, val in state_dict.items():
        if key.startswith("actor."):
            actor_state[key[len("actor."):]] = val
    result = actor.load_state_dict(actor_state, strict=False)
    if result.missing_keys:
        print(f"[WARNING] Missing actor keys: {result.missing_keys}")
    if result.unexpected_keys:
        print(f"[WARNING] Unexpected actor keys: {result.unexpected_keys}")
    actor.eval()

    # Extract empirical normalizer if present
    normalizer = None
    if "obs_rms.mean" in state_dict and "obs_rms.var" in state_dict:
        mean = state_dict["obs_rms.mean"]
        var = state_dict["obs_rms.var"]

        class EmpiricalNormalizer(nn.Module):
            def __init__(self, m: torch.Tensor, v: torch.Tensor) -> None:
                super().__init__()
                self.register_buffer("mean", m)
                self.register_buffer("std", torch.sqrt(v + 1e-8))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                return (x - self.mean) / self.std

        normalizer = EmpiricalNormalizer(mean, var)
        print("Embedded empirical normalization into ONNX model.")

    # Wrap and export
    export_model = PolicyExportWrapper(actor, normalizer)
    export_model.eval()

    dummy_input = torch.randn(1, num_observations)
    torch.onnx.export(
        export_model,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=["observations"],
        output_names=["actions"],
        dynamic_axes={
            "observations": {0: "batch_size"},
            "actions": {0: "batch_size"},
        },
    )

    print(f"Exported ONNX model to: {output_path}")
    print(f"  Input:  observations ({num_observations}D)")
    print(f"  Output: actions ({num_actions}D)")


def main() -> int:
    parser = argparse.ArgumentParser(description="Export tracking policy to ONNX.")
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--output", type=str, default="policy.onnx")
    parser.add_argument("--num_observations", type=int, default=189)
    parser.add_argument("--num_actions", type=int, default=29)
    parser.add_argument("--actor_hidden_dims", type=int, nargs="+", default=[512, 256, 128])
    parser.add_argument("--activation", type=str, default="elu")
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"Error: checkpoint not found: {args.checkpoint}")
        return 1

    export_policy_as_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_observations=args.num_observations,
        num_actions=args.num_actions,
        actor_hidden_dims=args.actor_hidden_dims,
        activation=args.activation,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
