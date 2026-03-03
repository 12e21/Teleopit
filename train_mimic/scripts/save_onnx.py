#!/usr/bin/env python3
"""Export trained policy to ONNX format for hardware deployment.

Adapted for Isaac Lab checkpoint format. Loads checkpoints from
logs/rsl_rl/{experiment_name}/{run_name}/model_*.pt and exports
the policy network to ONNX using the HardwareStudentFutureNN wrapper.

Usage:
    python save_onnx.py --checkpoint logs/rsl_rl/g1_mimic/run_001/model_5000.pt --output policy.onnx
"""

import argparse
import os
import torch
import torch.nn as nn

# Import policy network (pure PyTorch, works with both IsaacGym and Isaac Lab)
from train_mimic.rsl_rl.modules.actor_critic_mimic import ActorCriticMimic, get_activation


class HardwareStudentFutureNN(nn.Module):
    """Hardware deployment wrapper for student policy.
    
    This is a simplified wrapper that only exports the actor network
    for hardware deployment. It wraps the actor from ActorCriticMimic
    and handles observation normalization if a normalizer is provided.
    """
    
    def __init__(self, actor, normalizer=None):
        super().__init__()
        self.actor = actor
        self.normalizer = normalizer
    
    def forward(self, obs):
        """Forward pass with optional normalization."""
        if self.normalizer is not None:
            obs = self.normalizer.normalize(obs)
        return self.actor(obs)


def load_checkpoint(checkpoint_path: str) -> dict:
    """Load checkpoint from Isaac Lab format.
    
    Isaac Lab checkpoints may have different structures:
    - {'model_state_dict': ..., 'optimizer_state_dict': ..., 'iter': ...}
    - {'ac_state_dict': ..., 'optimizer_state_dict': ..., 'iter': ...}
    - Direct state dict (legacy format)
    """
    checkpoint = torch.load(checkpoint_path, map_location='cpu')
    return checkpoint


def export_policy_as_onnx(
    checkpoint_path: str,
    output_path: str,
    num_observations: int,
    num_actions: int,
    num_motion_observations: int = 9120,
    num_motion_steps: int = 10,
    motion_latent_dim: int = 64,
    actor_hidden_dims: list = None,
    activation: str = 'elu',
):
    """Export policy to ONNX format.
    
    Args:
        checkpoint_path: Path to checkpoint file
        output_path: Output ONNX file path
        num_observations: Total observation dimension (10098 for G1)
        num_actions: Action dimension (29 for G1)
        num_motion_observations: Motion observation dimension (9120 for G1)
        num_motion_steps: Number of motion timesteps (10 for G1)
        motion_latent_dim: Motion encoder latent dimension (64 for G1)
        actor_hidden_dims: Actor network hidden dimensions ([512,256,128] for G1)
        activation: Activation function ('elu' for G1)
    """
    if actor_hidden_dims is None:
        actor_hidden_dims = [512, 256, 128]
    
    # Load checkpoint
    checkpoint = load_checkpoint(checkpoint_path)
    
    # Extract policy state dict
    # Try multiple possible keys for Isaac Lab checkpoint format
    if 'model_state_dict' in checkpoint:
        policy_state_dict = checkpoint['model_state_dict']
    elif 'ac_state_dict' in checkpoint:
        policy_state_dict = checkpoint['ac_state_dict']
    else:
        # Assume checkpoint IS the state dict (legacy format)
        policy_state_dict = checkpoint
    
    # Instantiate policy network
    # Note: ActorCriticMimic is pure PyTorch, no Isaac Lab dependencies
    policy = ActorCriticMimic(
        num_observations=num_observations,
        num_critic_observations=num_observations,
        num_motion_observations=num_motion_observations,
        num_motion_steps=num_motion_steps,
        num_actions=num_actions,
        actor_hidden_dims=actor_hidden_dims,
        critic_hidden_dims=actor_hidden_dims,  # Same as actor
        motion_latent_dim=motion_latent_dim,
        activation=activation,
    )
    
    # Load weights
    policy.load_state_dict(policy_state_dict, strict=False)
    policy.eval()
    
    # Extract normalizer if available
    normalizer = checkpoint.get('normalizer', None)
    
    # Wrap with hardware deployment wrapper (actor only)
    hardware_policy = HardwareStudentFutureNN(policy.actor, normalizer)
    
    # Export to ONNX
    dummy_input = torch.randn(1, num_observations)
    torch.onnx.export(
        hardware_policy,
        dummy_input,
        output_path,
        export_params=True,
        opset_version=11,
        do_constant_folding=True,
        input_names=['observations'],
        output_names=['actions'],
        dynamic_axes={'observations': {0: 'batch_size'}, 'actions': {0: 'batch_size'}},
    )
    
    print(f"✓ Exported ONNX model to: {output_path}")
    print(f"  Input shape: (batch_size, {num_observations})")
    print(f"  Output shape: (batch_size, {num_actions})")


def main():
    parser = argparse.ArgumentParser(
        description="Export trained policy to ONNX format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Export with default G1 parameters
  python save_onnx.py --checkpoint logs/rsl_rl/g1_mimic/run_001/model_5000.pt
  
  # Export with custom output path
  python save_onnx.py --checkpoint model.pt --output my_policy.onnx
  
  # Export with custom dimensions
  python save_onnx.py --checkpoint model.pt --num_observations 10098 --num_actions 29
        """
    )
    parser.add_argument("--checkpoint", type=str, required=True, 
                        help="Path to checkpoint file (e.g., logs/rsl_rl/g1_mimic/run_001/model_5000.pt)")
    parser.add_argument("--output", type=str, default="policy.onnx", 
                        help="Output ONNX file path (default: policy.onnx)")
    parser.add_argument("--num_observations", type=int, default=10098, 
                        help="Observation dimension (default: 10098 for G1)")
    parser.add_argument("--num_actions", type=int, default=29, 
                        help="Action dimension (default: 29 for G1)")
    parser.add_argument("--num_motion_observations", type=int, default=9120,
                        help="Motion observation dimension (default: 9120 for G1)")
    parser.add_argument("--num_motion_steps", type=int, default=10,
                        help="Number of motion timesteps (default: 10 for G1)")
    parser.add_argument("--motion_latent_dim", type=int, default=64,
                        help="Motion encoder latent dimension (default: 64 for G1)")
    parser.add_argument("--actor_hidden_dims", type=int, nargs='+', default=[512, 256, 128],
                        help="Actor network hidden dimensions (default: 512 256 128)")
    parser.add_argument("--activation", type=str, default='elu',
                        help="Activation function (default: elu)")
    
    args = parser.parse_args()
    
    # Validate checkpoint exists
    if not os.path.exists(args.checkpoint):
        print(f"✗ Error: Checkpoint file not found: {args.checkpoint}")
        return 1
    
    export_policy_as_onnx(
        checkpoint_path=args.checkpoint,
        output_path=args.output,
        num_observations=args.num_observations,
        num_actions=args.num_actions,
        num_motion_observations=args.num_motion_observations,
        num_motion_steps=args.num_motion_steps,
        motion_latent_dim=args.motion_latent_dim,
        actor_hidden_dims=args.actor_hidden_dims,
        activation=args.activation,
    )
    
    return 0


if __name__ == "__main__":
    exit(main())
