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

import code
import torch
import torch.nn as nn
from torch.distributions import Normal
from torch.nn.modules import rnn
from torch.nn.modules.activation import ReLU


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

class Actor(nn.Module):
    def __init__(self, num_observations,
                 num_motion_observations,
                 num_motion_steps,
                 motion_latent_dim,
                 num_actions,
                 actor_hidden_dims, 
                 activation, 
                 layer_norm=False,
                 tanh_encoder_output=False, **kwargs) -> None:
        super().__init__()
        self.num_observations = num_observations
        self.num_actions = num_actions
        self.num_motion_observations = num_motion_observations
        self.num_motion_steps = num_motion_steps
        self.num_single_motion_observations = int(num_motion_observations / num_motion_steps)
        
        self.motion_encoder = MotionEncoder(activation, self.num_single_motion_observations, self.num_motion_steps, motion_latent_dim)

        actor_layers = []
        actor_layers.append(nn.Linear(self.num_observations - self.num_motion_observations + motion_latent_dim + self.num_single_motion_observations, 
                                      actor_hidden_dims[0]))
        actor_layers.append(activation)
        for l in range(len(actor_hidden_dims)):
            if l == len(actor_hidden_dims) - 1:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], num_actions))
            else:
                actor_layers.append(nn.Linear(actor_hidden_dims[l], actor_hidden_dims[l + 1]))
                if layer_norm and l == len(actor_hidden_dims) - 2:
                    actor_layers.append(nn.LayerNorm(actor_hidden_dims[l + 1]))
                actor_layers.append(activation)
        if tanh_encoder_output:
            actor_layers.append(nn.Tanh())
        self.actor_backbone = nn.Sequential(*actor_layers)

    def forward(self, obs, hist_encoding: bool = False):
        motion_obs = obs[:, :self.num_motion_observations]
        motion_latent = self.motion_encoder(motion_obs)
        backbone_input = torch.cat([obs[:, self.num_motion_observations:], obs[:, :self.num_single_motion_observations], motion_latent], dim=1)
        backbone_output = self.actor_backbone(backbone_input)
        return backbone_output


class ActorCriticMimic(nn.Module):
    is_recurrent = False
    def __init__(self,  
                num_observations,
                num_critic_observations,
                num_motion_observations,
                num_motion_steps,
                num_actions,
                actor_hidden_dims=[256, 256, 256],
                critic_hidden_dims=[256, 256, 256],
                motion_latent_dim=64,
                activation='elu',
                init_noise_std=1.0,
                fix_action_std=False,
                action_std=None,
                layer_norm=False,
                **kwargs):
        if kwargs:
            print("ActorCritic.__init__ got unexpected arguments, which will be ignored: " + str([key for key in kwargs.keys()]))
        super().__init__()

        self.fix_action_std = fix_action_std
        
        self.kwargs = kwargs
        activation = get_activation(activation)
        
        self.actor = Actor(num_observations=num_observations, # all priv info
                           num_actions=num_actions, 
                           num_motion_observations=num_motion_observations,
                           num_motion_steps=num_motion_steps,
                           motion_latent_dim=motion_latent_dim,
                           actor_hidden_dims=actor_hidden_dims, 
                           activation=activation, layer_norm=layer_norm, tanh_encoder_output=False)
        
        self.num_motion_observations = num_motion_observations
        self.num_single_motion_obs = int(num_motion_observations / num_motion_steps)
        
        # Value function
        self.critic_motion_encoder = MotionEncoder(activation, self.num_single_motion_obs, num_motion_steps, motion_latent_dim)
        critic_layers = []
        critic_layers.append(nn.Linear(num_critic_observations - num_motion_observations + motion_latent_dim + self.num_single_motion_obs, critic_hidden_dims[0]))
        critic_layers.append(activation)
        for l in range(len(critic_hidden_dims)):
            if l == len(critic_hidden_dims) - 1:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], 1))
            else:
                critic_layers.append(nn.Linear(critic_hidden_dims[l], critic_hidden_dims[l + 1]))
                if layer_norm and l == len(critic_hidden_dims) - 2:
                    critic_layers.append(nn.LayerNorm(critic_hidden_dims[l + 1]))
                critic_layers.append(activation)
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
    # not used at the moment
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
        if not eval:
            actions_mean = self.actor(observations, eval)
            return actions_mean
        else:
            actions_mean = self.actor(observations, eval=True)
            return actions_mean

    def evaluate(self, critic_observations, **kwargs):
        motion_obs = critic_observations[:, :self.num_motion_observations]
        motion_single_obs = critic_observations[:, :self.num_single_motion_obs]
        motion_latent = self.critic_motion_encoder(motion_obs)
        backbone_input = torch.cat([critic_observations[:, self.num_motion_observations:], motion_single_obs, motion_latent], dim=1)
        value = self.critic(backbone_input)
        return value
    
    def reset_std(self, std, num_actions, device):
        new_std = std * torch.ones(num_actions, device=device)
        self.std.data = new_std.data
        
    def if_fix_std(self):
        return self.fix_action_std

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
