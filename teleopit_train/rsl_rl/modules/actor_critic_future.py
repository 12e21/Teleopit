# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
# 
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
# list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
# this list of conditions and the following disclaimer in the documentation
# and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
# contributors may be used to endorse or promote products derived from
# this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU


def get_activation(act_name):
    if act_name == "elu":
        return nn.ELU()
    elif act_name == "selu":
        return nn.SELU()
    elif act_name == "relu":
        return nn.ReLU()
    elif act_name == "crelu":
        return nn.ReLU()
    elif act_name == "lrelu":
        return nn.LeakyReLU()
    elif act_name == "tanh":
        return nn.Tanh()
    elif act_name == "sigmoid":
        return nn.Sigmoid()
    elif act_name == "silu":
        return nn.SiLU()
    else:
        print("invalid activation function!")
        return None

class MotionEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super().__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 20

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        elif tsteps == 1:
            self.conv_layers = nn.Flatten()
        else:
            raise(ValueError("tsteps must be 1, 10, 20 or 50"))
        self.linear_output = nn.Linear(channel_size * 3, output_size)

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output


class HistoryEncoder(nn.Module):
    def __init__(self, activation_fn, input_size, tsteps, output_size, tanh_encoder_output=False):
        super().__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps

        channel_size = 20

        self.encoder = nn.Sequential(
                nn.Linear(input_size, 3 * channel_size), self.activation_fn,
                )

        if tsteps == 50:
            self.conv_layers = nn.Sequential(
                    nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 8, stride = 4), self.activation_fn,
                    nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn,
                    nn.Conv1d(in_channels = channel_size, out_channels = channel_size, kernel_size = 5, stride = 1), self.activation_fn, nn.Flatten())
        elif tsteps == 10:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 2, stride = 1), self.activation_fn,
                nn.Flatten())
        elif tsteps == 20:
            self.conv_layers = nn.Sequential(
                nn.Conv1d(in_channels = 3 * channel_size, out_channels = 2 * channel_size, kernel_size = 6, stride = 2), self.activation_fn,
                nn.Conv1d(in_channels = 2 * channel_size, out_channels = channel_size, kernel_size = 4, stride = 2), self.activation_fn,
                nn.Flatten())
        elif tsteps == 1:
            self.conv_layers = nn.Flatten()
        else:
            assert False, f"tsteps must be 1, 10, 20 or 50, but got {tsteps}"
        self.linear_output = nn.Linear(channel_size * 3, output_size)

    def forward(self, obs):
        nd = obs.shape[0]
        T = self.tsteps
        projection = self.encoder(obs.reshape([nd * T, -1])) # do projection for n_proprio -> 32
        output = self.conv_layers(projection.reshape([nd, T, -1]).permute((0, 2, 1)))
        output = self.linear_output(output)
        return output
 

class FutureMotionEncoder(nn.Module):
    """Simplified encoder for future motion observations without attention."""
    
    def __init__(self, activation_fn, input_size, tsteps, output_size, 
                 attention_heads=4, dropout=0.1, temporal_embedding_dim=64):
        super().__init__()
        self.activation_fn = activation_fn
        self.tsteps = tsteps
        self.input_size = input_size
        self.output_size = output_size
        
        # Simple approach: flatten all future observations and use MLP
        total_input_size = input_size * tsteps  # Flatten future observations (ignore mask for now)
        
        # Simple MLP encoder
        self.encoder = nn.Sequential(
            nn.Linear(total_input_size, 256),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(256, 128),
            activation_fn,
            nn.Dropout(dropout),
            nn.Linear(128, output_size)
        )
        
        # Initialize with stable weights
        for layer in self.encoder:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight, gain=0.5)
                nn.init.zeros_(layer.bias)
        
    def forward(self, obs):
        """
        Args:
            obs: (batch_size, tsteps, input_size + 1)  # +1 for mask indicator
        """
        batch_size = obs.shape[0]
        
        # Separate mask indicator from observations
        future_obs = obs[:, :, :-1]  # (batch_size, tsteps, input_size)
        mask_indicator = obs[:, :, -1]  # (batch_size, tsteps)
        
        # Simple approach: flatten and encode
        # For now, ignore masking and just use all future observations
        flattened = future_obs.reshape(batch_size, -1)  # (batch_size, tsteps * input_size)
        
        # Encode
        output = self.encoder(flattened)  # (batch_size, output_size)
        
        return output


class ActorFuture(nn.Module):
    """Actor network with future motion support."""
    
    def __init__(self, num_observations,
                 num_motion_observations,
                 num_priop_observations,
                 num_motion_steps,
                 num_future_observations,
                 num_future_steps,
                 motion_latent_dim,
                 future_latent_dim,
                 num_actions,
                 actor_hidden_dims, 
                 activation, 
                 history_latent_dim,
                 num_history_steps,
                 layer_norm=False,
                 future_encoder_dims=[256, 256, 128],
                 future_attention_heads=4,
                 future_dropout=0.1,
                 temporal_embedding_dim=64,
                 use_history_encoder=True,
                 use_motion_encoder=True,
                 tanh_encoder_output=False, **kwargs):
        """
        observation structure: motion_obs, priop_obs, history_obs, future_obs
        """
        super().__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_motion_observations = num_motion_observations
        self.num_priop_observations = num_priop_observations
        self.num_motion_steps = num_motion_steps
        self.num_history_steps = num_history_steps
        self.num_future_observations = num_future_observations
        self.num_future_steps = num_future_steps
        self.use_history_encoder = use_history_encoder
        self.use_motion_encoder = use_motion_encoder
        
        # Calculate single step sizes
        self.num_single_motion_observations = int(num_motion_observations / num_motion_steps)
        self.num_single_priop_observations = num_priop_observations
        
        # Calculate history size
        # The history observations are simply the single step size repeated num_history_steps times
        # We don't need to calculate from total observations as there might be additional components like masked_priv_info
        num_single_history_observations = num_motion_observations + num_priop_observations
        history_size = num_single_history_observations * num_history_steps
        
        # Calculate future single step size
        self.num_single_future_observations = int(num_future_observations / num_future_steps) if num_future_observations > 0 else 0
        self.future_latent_dim = future_latent_dim
        
        # Motion encoder (same as tracking)
        if self.use_motion_encoder:
            self.motion_encoder = MotionEncoder(activation, self.num_single_motion_observations, self.num_motion_steps, motion_latent_dim)
        else:
            self.motion_encoder = nn.Identity()
            motion_latent_dim = self.num_single_motion_observations
        
        # History encoder (same as tracking)
        if self.use_history_encoder:
            self.history_encoder = HistoryEncoder(activation, num_single_history_observations, self.num_history_steps, history_latent_dim)
        else:
            self.history_encoder = nn.Identity()
            history_latent_dim = history_size
            
        # Future motion encoder (new)
        if self.num_single_future_observations > 0:
            self.future_encoder = FutureMotionEncoder(
                activation, 
                self.num_single_future_observations - 1,  # -1 because mask indicator is separate
                self.num_future_steps, 
                future_latent_dim,
                attention_heads=future_attention_heads,
                dropout=future_dropout,
                temporal_embedding_dim=temporal_embedding_dim
            )
        else:
            self.future_encoder = None
        
        # Main actor network
        input_dim = (motion_latent_dim + self.num_single_motion_observations + 
                    self.num_single_priop_observations + history_latent_dim + future_latent_dim)
        
        actor_layers = []
        first_layer = nn.Linear(input_dim, actor_hidden_dims[0])
        # Initialize first layer with smaller weights
        nn.init.xavier_uniform_(first_layer.weight, gain=0.5)
        nn.init.zeros_(first_layer.bias)
        actor_layers.append(first_layer)
        actor_layers.append(activation)
        
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                final_layer = nn.Linear(actor_hidden_dims[l], num_actions)
                # Initialize final layer with very small weights
                nn.init.xavier_uniform_(final_layer.weight, gain=0.1)
                nn.init.zeros_(final_layer.bias)
                actor_layers.append(final_layer)
            else:
                layer = nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1])
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
                actor_layers.append(layer)
                if layer_norm and l == len(actor_hidden_dims) - 2:
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
                
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
            
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, hist_encoding: bool = False):
        # Parse observations based on structure: current + history + future
        # current = motion + priop
        current_size = self.num_motion_observations + self.num_priop_observations
        
        # Extract current observations
        motion_obs = obs[:, :self.num_motion_observations]
        single_motion_obs = obs[:, :self.num_single_motion_observations]
        priop_obs = obs[:, self.num_motion_observations:current_size]
        
        # Extract history observations
        history_start = current_size
        history_size = self.num_history_steps * current_size
        history_end = history_start + history_size
        history_obs = obs[:, history_start:history_end]
        
        # Extract future observations (excluding any additional components like masked_priv_info)
        future_start = history_end
        future_end = future_start + self.num_future_observations
        future_obs = obs[:, future_start:future_end]
        
        # Encode all components
        motion_latent = self.motion_encoder(motion_obs)
        history_latent = self.history_encoder(history_obs)
        
        # Encode future motion - simplified approach
        if self.future_encoder is not None and self.num_future_observations > 0 and future_obs.shape[1] > 0:
            # Reshape future observations to (batch_size, num_future_steps, obs_per_step)
            future_obs_reshaped = future_obs.reshape(-1, self.num_future_steps, self.num_single_future_observations)
            future_latent = self.future_encoder(future_obs_reshaped)
        else:
            # Create dummy future latent if no future observations
            future_latent = torch.zeros(obs.shape[0], self.future_latent_dim, device=obs.device)
        
        # Combine all features
        backbone_input = torch.cat([
            single_motion_obs, 
            priop_obs, 
            motion_latent, 
            history_latent, 
            future_latent
        ], dim=1)
        
        backbone_output = self.actor_backbone(backbone_input)
        return backbone_output


class ActorCriticFuture(nn.Module):
    """Actor-Critic network with future motion support."""
    
    is_recurrent = False
    
    def __init__(self,  
                num_observations,
                num_critic_observations,
                num_motion_observations,
                num_motion_steps,
                num_priop_observations,
                num_history_steps,
                num_actions,
                actor_hidden_dims=[256, 256, 256],
                critic_hidden_dims=[256, 256, 256],
                motion_latent_dim=128,
                history_latent_dim=128,
                future_latent_dim=128,
                activation='silu',
                init_noise_std=1.0,
                fix_action_std=False,
                action_std=None,
                layer_norm=False,
                # Future motion specific parameters
                future_encoder_dims=[256, 256, 128],
                future_attention_heads=4,
                future_dropout=0.1,
                temporal_embedding_dim=64,
                **kwargs):
        if kwargs:
            print("ActorCriticFuture.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        self.fix_action_std = fix_action_std
        self.kwargs = kwargs
        activation_fn = get_activation(activation)
        
        # Calculate future motion dimensions based on environment config
        # Observation structure: current + history + future [+ masked_priv_info]
        # current = motion + priop
        # history = (motion + priop) * history_steps
        # future = future_observations
        
        single_obs_size = num_motion_observations + num_priop_observations
        expected_history_size = num_history_steps * single_obs_size
        expected_current_size = single_obs_size
        
        # Use explicit num_future_observations if provided in kwargs, otherwise calculate
        if 'num_future_observations' in kwargs:
            num_future_observations = kwargs['num_future_observations']
            print(f"ActorCriticFuture: Using explicit num_future_observations = {num_future_observations}")
        else:
            num_future_observations = max(0, num_observations - expected_current_size - expected_history_size)
            print(f"ActorCriticFuture: Calculated num_future_observations = {num_future_observations}")
        
        # Get future steps from kwargs if available, otherwise use default
        num_future_steps = kwargs.get('num_future_steps', 10)
        
        print(f"ActorCriticFuture: obs={num_observations}, motion={num_motion_observations}, priop={num_priop_observations}")
        print(f"ActorCriticFuture: current={expected_current_size}, history={expected_history_size}, future={num_future_observations}")
        print(f"ActorCriticFuture: future_steps={num_future_steps}")
        
        # Actor network
        self.actor = ActorFuture(
            num_observations=num_observations,
            num_actions=num_actions, 
            num_motion_observations=num_motion_observations,
            num_priop_observations=num_priop_observations,
            num_motion_steps=num_motion_steps,
            num_future_observations=num_future_observations,
            num_future_steps=num_future_steps,
            num_history_steps=num_history_steps,
            motion_latent_dim=motion_latent_dim,
            future_latent_dim=future_latent_dim,
            history_latent_dim=history_latent_dim,
            actor_hidden_dims=actor_hidden_dims, 
            activation=activation_fn,
            layer_norm=layer_norm,
            future_encoder_dims=future_encoder_dims,
            future_attention_heads=future_attention_heads,
            future_dropout=future_dropout,
            temporal_embedding_dim=temporal_embedding_dim,
            # use_history_encoder=False,
            # use_motion_encoder=False,
            use_history_encoder=True,
            use_motion_encoder=True,
            tanh_encoder_output=kwargs.get('tanh_encoder_output', False)
        )
        # count the number of parameters in the actor network
        print(f"ActorCriticFuture: actor network has {sum(p.numel() for p in self.actor.parameters()) / 1e6:.2f}M parameters")
        
        # Store dimensions for critic
        self.num_motion_observations = num_motion_observations
        self.num_single_motion_obs = int(num_motion_observations / num_motion_steps)
        
        # Critic network (uses privileged observations without future motion)
        self.critic_motion_encoder = MotionEncoder(activation_fn, self.num_single_motion_obs, num_motion_steps, motion_latent_dim)
        
        critic_input_dim = num_critic_observations - num_motion_observations + motion_latent_dim + self.num_single_motion_obs
        
        critic_layers = []
        critic_layers.append(nn.Linear(critic_input_dim, critic_hidden_dims[0]))
        critic_layers.append(activation_fn)
        
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                if layer_norm and l == len(critic_hidden_dims) - 2:
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[l + 1]))
                critic_layers.append(activation_fn)
                
        self.critic = nn.Sequential(*critic_layers)

        # Action noise
        if self.fix_action_std:
            self.init_action_std_tensor = torch.tensor(action_std)
            self.std = nn.Parameter(self.init_action_std_tensor, requires_grad=False)
        else:
            self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            
        self.distribution = None
        # disable args validation for speedup
        Normal.set_default_validate_args = False
        
    @staticmethod
    def init_weights(sequential, scales):
        [torch.nn.init.orthogonal_(module.weight, gain=scales[idx]) for idx, module in
         enumerate(mod for mod in sequential if isinstance(mod, nn.Linear))]

    def reset(self, dones=None):
        pass

    def forward(self):
        raise NotImplementedError
    
    @property
    def action_mean(self):
        return self.distribution.mean

    @property
    def action_std(self):
        return self.distribution.stddev
    
    @property
    def entropy(self):
        return self.distribution.entropy().sum(dim=-1)

    def update_distribution(self, observations):
        mean = self.actor(observations)
        self.distribution = Normal(mean, mean*0. + self.std)

    def act(self, observations, **kwargs):
        self.update_distribution(observations)
        return self.distribution.sample()
    
    def get_actions_log_prob(self, actions):
        return self.distribution.log_prob(actions).sum(dim=-1)

    def act_inference(self, observations, eval=False, **kwargs):
        actions_mean = self.actor(observations)
        return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        # Critic uses privileged observations (no future motion)
        motion_obs = critic_observations[:, :self.num_motion_observations]
        motion_single_obs = critic_observations[:, :self.num_single_motion_obs]
        motion_latent = self.critic_motion_encoder(motion_obs)
        
        backbone_input = torch.cat([
            critic_observations[:, self.num_motion_observations:], 
            motion_single_obs, 
            motion_latent
        ], dim=1)
        
        value = self.critic(backbone_input)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data
        
    def if_fix_std(self):
        return self.fix_action_std