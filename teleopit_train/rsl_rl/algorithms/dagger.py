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

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from teleopit_train.rsl_rl.modules import ActorCritic, DAggerActor
from teleopit_train.rsl_rl.storage import RolloutStorage, ReplayBuffer
from teleopit_train.rsl_rl.utils import unpad_trajectories
import time


class DAgger:
    def __init__(self,
                 env, 
                 teacher_actor,
                 dagger_actor,
                 num_learning_epochs=1,
                 num_mini_batches=1,
                 learning_rate=1e-3,
                 max_grad_norm=1.0,
                 device='cpu',
                 **kwargs
                 ):
        
        self.env = env
        self.device = device
        
        self.learning_rate = learning_rate
        
        # teacher actor
        self.teacher_actor = teacher_actor
        
        # dagger actor
        self.dagger_actor = dagger_actor
        self.dagger_actor.to(self.device)
        self.storage = None
        self.optimizer = optim.Adam(self.dagger_actor.parameters(), lr=self.learning_rate)
        self.transition = RolloutStorage.Transition()
        
        self.num_learning_epochs = num_learning_epochs
        self.num_mini_batches = num_mini_batches
        self.max_grad_norm = max_grad_norm
        
        self.counter = 0
        
    def init_storage(self, num_envs, num_transitions_per_env, actor_obs_shape, critic_obs_shape, action_shape):
        self.storage = RolloutStorage(num_envs, num_transitions_per_env, actor_obs_shape,  critic_obs_shape, action_shape, self.device)
        
    def test_mode(self):
        self.dagger_actor.eval()
    
    def train_mode(self):
        self.dagger_actor.train()
        
    def act(self, obs, critic_obs, dummy=False):
        # Compute the actions and values, use proprio to compute estimated priv_states then actions, but store true priv_states
        self.transition.actions = self.dagger_actor(obs).detach()
        self.transition.observations = obs
        self.transition.critic_observations = critic_obs
        
        self.transition.values = torch.zeros(self.transition.actions.shape[0], 1, device=self.device)
        self.transition.actions_log_prob = torch.zeros(self.transition.actions.shape[0], 1, device=self.device)
        self.transition.action_mean = torch.zeros_like(self.transition.actions)
        self.transition.action_sigma = torch.zeros_like(self.transition.actions)
        
        return self.transition.actions
    
    def process_env_step(self, rewards, done, infos):
        self.transition.rewards = rewards.clone()
        self.transition.dones = done
        
        self.storage.add_transitions(self.transition)
        self.transition.clear()
        
        return rewards
    
    def update(self):
        mean_l2_loss = 0
        
        generator = self.storage.mini_batch_generator(self.num_mini_batches, self.num_learning_epochs)
        
        for sample in generator:
            obs_batch, critic_obs_batch, actions_batch, _, _, _, _, \
                _, _, _, _ = sample
            
            # Compute the loss
            student_actions = self.dagger_actor(obs_batch)
            with torch.no_grad():
                teacher_actions = self.teacher_actor(critic_obs_batch)
            
            action_loss = (student_actions - teacher_actions).pow(2).mean()
            
            self.optimizer.zero_grad()
            action_loss.backward()
            nn.utils.clip_grad_norm_(self.dagger_actor.parameters(), self.max_grad_norm)
            self.optimizer.step()
            
            mean_l2_loss += action_loss.item()
        
        num_updates = self.num_learning_epochs * self.num_mini_batches
        mean_l2_loss /= num_updates
        
        self.storage.clear()
        self.update_counter()
        
        return mean_l2_loss

    def update_counter(self):
        self.counter += 1
