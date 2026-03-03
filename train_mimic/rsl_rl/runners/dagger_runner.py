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

import time
import os
from collections import deque
import statistics

# from torch.utils.tensorboard import SummaryWriter
import torch
import torch.optim as optim
import wandb
# import ml_runlog
import datetime

import numpy as np
from train_mimic.rsl_rl.algorithms import *
from train_mimic.rsl_rl.modules import *
from train_mimic.rsl_rl.storage.replay_buffer import ReplayBuffer
from train_mimic.rsl_rl.env import VecEnv
import sys
from copy import copy, deepcopy
import warnings
from train_mimic.rsl_rl.utils.normalizer import Normalizer

from train_mimic import TRAIN_MIMIC_ROOT_DIR


def get_policy_path(proj_name, exptid, checkpoint=-1):
    policy_dir = os.path.join(TRAIN_MIMIC_ROOT_DIR, "logs", proj_name, exptid)
    if checkpoint == -1:
        models = [file for file in os.listdir(policy_dir) if "model" in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
    else:
        model = "model_{}.pt".format(checkpoint)
    
    return os.path.join(policy_dir, model)

class DAggerRunner:

    def __init__(self,
                 env: VecEnv,
                 train_cfg,
                 log_dir=None,
                 device='cpu', **kwargs):

        self.cfg=train_cfg["runner"]
        self.alg_cfg = train_cfg["algorithm"]
        self.policy_cfg = train_cfg["policy"]
        self.device = device
        self.env = env
        self.normalize_obs = env.cfg.env.normalize_obs
        
        self.teacher_cfg = train_cfg["teachercfg"]["runner"]
        self.teacher_policy_cfg = train_cfg["teachercfg"]["policy"]
        self.teacher_alg_cfg = train_cfg["teachercfg"]["algorithm"]

        self.warm_iters = self.cfg["warm_iters"]
        self.eval_student = self.cfg["eval_student"]

        teacher_policy_class = eval(self.teacher_cfg["policy_class_name"])
        dagger_policy_class = eval(self.cfg["policy_class_name"])
                
        self.teacher_actor_critic = teacher_policy_class(num_observations=self.env.num_privileged_obs,
                                    num_critic_observations=self.env.num_privileged_obs,
                                    num_motion_observations=self.env.cfg.env.n_priv_mimic_obs,
                                    num_motion_steps=len(self.env.cfg.env.tar_motion_steps_priv),
                                    num_actions=self.env.num_actions,
                                    **self.teacher_policy_cfg).to(self.device)
        if self.normalize_obs:
            self.teacher_normalizer = Normalizer(shape=self.env.num_privileged_obs, device=self.device, dtype=env.obs_buf.dtype)
        else:
            self.teacher_normalizer = None
        
        if not self.eval_student:
            teacher_policy_pth = get_policy_path(self.cfg["teacher_proj_name"], exptid=self.cfg["teacher_experiment_name"], checkpoint=self.cfg["teacher_checkpoint"])
            self.load_teacher(teacher_policy_pth)
        else:
            print("Evaluating student policy only, not loading teacher policy.")
        
        self.teacher_actor = self.get_teacher_inference_policy(device=self.device)
        student_actor = dagger_policy_class(num_observations=self.env.num_obs,
                                            num_actions=self.env.num_actions,
                                            num_hist=self.env.cfg.env.history_len,
                                            **self.policy_cfg).to(self.device)
        if self.normalize_obs:
            self.student_normalizer = Normalizer(shape=self.env.num_obs, device=self.device, dtype=env.obs_buf.dtype)
        else:
            self.student_normalizer = None
            
        alg_class = eval(self.cfg["algorithm_class_name"]) # DAgger
        self.alg = alg_class(self.env,
                                  self.teacher_actor, # teacher actor
                                  student_actor, # student actor
                                  device=self.device, **self.alg_cfg)
        self.num_steps_per_env = self.cfg["num_steps_per_env"]
        self.save_interval = self.cfg["save_interval"]
        
        self.alg.init_storage(
            self.env.num_envs, 
            self.num_steps_per_env, 
            [self.env.num_obs], 
            [self.env.num_privileged_obs], 
            [self.env.num_actions],
        )

        self.learn = self.learn_DAgger
            
        # Log
        self.log_dir = log_dir
        self.writer = None
        self.tot_timesteps = 0
        self.tot_time = 0
        self.current_learning_iteration = 0
        
    def learn_DAgger(self, num_learning_iterations, init_at_random_ep_len=False):
        mean_dagger_loss = 0.
        
        obs = self.env.get_observations()
        privileged_obs = self.env.get_privileged_observations()
        critic_obs = privileged_obs if privileged_obs is not None else obs
        obs, critic_obs = obs.to(self.device), critic_obs.to(self.device)
        if self.normalize_obs:
            critic_obs = self.teacher_normalizer.normalize(critic_obs)
            obs = self.student_normalizer.normalize(obs)
        infos = {}
        self.alg.dagger_actor.train()
        
        ep_infos = []
        rewbuffer = deque(maxlen=100)
        lenbuffer = deque(maxlen=100)
        cur_reward_sum = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        cur_episode_length = torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device)
        
        tot_iter = self.current_learning_iteration + num_learning_iterations
        self.start_learning_iteration = copy(self.current_learning_iteration)
        
        
        for it in range(self.current_learning_iteration, tot_iter):
            start = time.time()
            with torch.inference_mode():
                for i in range(self.num_steps_per_env):
                    if it < self.warm_iters:
                        actions = self.teacher_actor(critic_obs)
                        self.alg.act(obs, critic_obs)
                    else:
                        actions = self.alg.act(obs, critic_obs)
                    obs, privileged_obs, rewards, dones, infos = self.env.step(actions)
                    critic_obs = privileged_obs if privileged_obs is not None else obs
                    obs, critic_obs, rewards, dones = obs.to(self.device), critic_obs.to(self.device), rewards.to(self.device), dones.to(self.device)
                    
                    if self.normalize_obs:
                        before_norm_obs = obs.clone()
                        critic_obs = self.teacher_normalizer.normalize(critic_obs)
                        obs = self.student_normalizer.normalize(obs)
                        if self._need_normalizer_update(it, self.alg_cfg["normalizer_update_iterations"]):
                            self.student_normalizer.record(before_norm_obs)

                    total_rew = self.alg.process_env_step(rewards, dones, infos)
                    
                    if self.log_dir is not None:
                        if 'episode' in infos:
                            ep_infos.append(infos['episode'])
                        cur_reward_sum += total_rew
                        cur_episode_length += 1
                        new_ids = (dones > 0).nonzero(as_tuple=False)
                        rewbuffer.extend(cur_reward_sum[new_ids][:, 0].cpu().numpy().tolist())
                        lenbuffer.extend(cur_episode_length[new_ids][:, 0].cpu().numpy().tolist())
                        cur_reward_sum[new_ids] = 0
                        cur_episode_length[new_ids] = 0
                stop = time.time()
                collection_time = stop - start
                
                # Learning step
                start = stop
        
            average_episode_length = average_episode_length = torch.mean(self.env.episode_length.float()).item() if hasattr(self.env, "episode_length") else 0
            
            mean_dagger_loss = self.alg.update()
            
            stop = time.time()
            learn_time = stop - start
            if self.log_dir is not None:
                self.log(locals())
            
            if it < 2500:
                if it % self.save_interval == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            else:
                if it % (2*self.save_interval) == 0:
                    self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(it)))
            
            ep_infos.clear()
        
        self.save(os.path.join(self.log_dir, 'model_{}.pt'.format(self.current_learning_iteration)))
    
    def log(self, locs, width=80, pad=35):
        self.tot_timesteps += self.num_steps_per_env * self.env.num_envs
        self.tot_time += locs['collection_time'] + locs['learn_time']
        iteration_time = locs['collection_time'] + locs['learn_time']

        ep_string = f''
        wandb_dict = {}
        if locs['ep_infos']:
            for key in locs['ep_infos'][0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in locs['ep_infos']:
                    # handle scalar and zero dimensional tensor infos
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                value = torch.mean(infotensor)
                if "metric" in key:
                    wandb_dict['Episode_rew_metrics/' + key] = value
                else:
                    if "tracking" in key:
                        wandb_dict['Episode_rew_tracking/' + key] = value
                    elif "curriculum" in key:
                        wandb_dict['Episode_curriculum/' + key] = value
                    else:
                        wandb_dict['Episode_rew_regularization/' + key] = value
                    ep_string += f"""{f'Mean episode {key}:':>{pad}} {value:.4f}\n""" # dont print metrics
        fps = int(self.num_steps_per_env * self.env.num_envs / (locs['collection_time'] + locs['learn_time']))
        
        wandb_dict['Loss/dagger_loss'] = locs['mean_dagger_loss']

        wandb_dict['Perf/total_fps'] = fps
        wandb_dict['Perf/collection time'] = locs['collection_time']
        wandb_dict['Perf/learning_time'] = locs['learn_time']
        if len(locs['rewbuffer']) > 0:
            wandb_dict['Train/mean_reward'] = statistics.mean(locs['rewbuffer'])
            wandb_dict['Train/mean_episode_length'] = statistics.mean(locs['lenbuffer'])
            
        wandb.log(wandb_dict, step=locs['it'])

        str = f" \033[1m Learning iteration {locs['it']}/{self.current_learning_iteration + locs['num_learning_iterations']} \033[0m "

        average_episode_length = f"""{'Average_episode_length:':>{pad}} {locs['average_episode_length']:.4f}\n"""
        if len(locs['rewbuffer']) > 0:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Experiment Name:':>{pad}} {os.path.basename(self.log_dir)}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'Mean reward (total):':>{pad}} {statistics.mean(locs['rewbuffer']):.2f}\n"""
                          f"""{'Mean episode length:':>{pad}} {statistics.mean(locs['lenbuffer']):.2f}\n""")
        else:
            log_string = (f"""{'#' * width}\n"""
                          f"""{str.center(width, ' ')}\n\n"""
                          f"""{'Computation:':>{pad}} {fps:.0f} steps/s (collection: {locs[
                            'collection_time']:.3f}s, learning {locs['learn_time']:.3f}s)\n"""
                          f"""{'DAgger loss:':>{pad}} {locs['mean_dagger_loss']:.4f}\n""")

        log_string += f"""{'-' * width}\n"""
        log_string += ep_string
        log_string += f"""{'-' * width}\n"""
        log_string += average_episode_length
        curr_it = locs['it'] - self.start_learning_iteration
        eta = self.tot_time / (curr_it + 1) * (locs['num_learning_iterations'] - curr_it)
        mins = eta // 60
        secs = eta % 60
        log_string += (f"""{'-' * width}\n"""
                       f"""{'Total timesteps:':>{pad}} {self.tot_timesteps}\n"""
                       f"""{'Iteration time:':>{pad}} {iteration_time:.2f}s\n"""
                       f"""{'Total time:':>{pad}} {self.tot_time:.2f}s\n"""
                       f"""{'ETA:':>{pad}} {mins:.0f} mins {secs:.1f} s\n""")
        print(log_string)

    def save(self, path, infos=None):
        if self.normalize_obs:
            state_dict = {
            'model_state_dict': self.alg.dagger_actor.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'normalizer': self.student_normalizer,
            'infos': infos,
            }
        else:
            state_dict = {
            'model_state_dict': self.alg.dagger_actor.state_dict(),
            'optimizer_state_dict': self.alg.optimizer.state_dict(),
            'iter': self.current_learning_iteration,
            'infos': infos,
            }
        torch.save(state_dict, path)
    
    def load_teacher(self, path):
        print("*" * 80)
        print("Loading teacher policy from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.teacher_actor_critic.load_state_dict(loaded_dict['model_state_dict'])
        if self.normalize_obs:
            self.teacher_normalizer = loaded_dict['normalizer']
        print("*" * 80)
    
    def get_teacher_inference_policy(self, device=None):
        self.teacher_actor_critic.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.teacher_actor_critic.to(device)
        return self.teacher_actor_critic.act_inference

    def load(self, path, load_optimizer=True):
        print("*" * 80)
        print("Loading model from {}...".format(path))
        loaded_dict = torch.load(path, map_location=self.device)
        self.alg.dagger_actor.load_state_dict(loaded_dict['model_state_dict'])
        if self.normalize_obs:
            self.student_normalizer = loaded_dict['normalizer']
        if load_optimizer:
            self.alg.optimizer.load_state_dict(loaded_dict['optimizer_state_dict'])
        self.current_learning_iteration = int(os.path.basename(path).split("_")[1].split(".")[0])
        print("*" * 80)
        return loaded_dict['infos']
    
    def get_inference_policy(self, device=None):
        self.alg.dagger_actor.eval() # switch to evaluation mode (dropout for example)
        if device is not None:
            self.alg.dagger_actor.to(device)
        return self.alg.dagger_actor
    
    def _need_normalizer_update(self, iterations, update_iterations):
        return iterations < update_iterations

    def get_normalizer(self, device=None):
        if device is not None:
            self.student_normalizer.to(device)
        return self.student_normalizer