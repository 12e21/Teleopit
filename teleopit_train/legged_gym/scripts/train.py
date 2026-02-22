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

import os
from datetime import datetime

import isaacgym
from teleopit_train.legged_gym.envs import *
from teleopit_train.legged_gym.gym_utils import get_args, task_registry

import torch
import wandb

def train(args):
    args.headless = True
    
    log_pth = TELEOPIT_TRAIN_ROOT_DIR + "/logs/{}/".format(args.proj_name) + args.exptid
    try:
        os.makedirs(log_pth)
    except:
        pass
    
    if args.debug:
        mode = "disabled"
        args.rows = 10
        args.cols = 5
        args.num_envs = 4
        args.headless = False
        # args.headless = True
    else:
        mode = "online"
    
    if args.no_wandb:
        mode = "disabled"
        
    robot_type = args.task.split("_")[0]
    
    try:
        wandb.init(project="twist", name=args.exptid, mode=mode, dir="../../logs")
    except:
        wandb.init(project="g1_mimic", name=args.exptid, mode=mode, dir="../../logs")
    # wandb.save(TELEOPIT_TRAIN_ENVS_DIR + "/base/legged_robot_config.py", policy="now")
    # wandb.save(TELEOPIT_TRAIN_ENVS_DIR + "/base/legged_robot.py", policy="now")
    # wandb.save(TELEOPIT_TRAIN_ENVS_DIR + "/base/humanoid_config.py", policy="now")
    # wandb.save(TELEOPIT_TRAIN_ENVS_DIR + "/base/humanoid.py", policy="now")
    if robot_type == "g1":
        wandb.save(TELEOPIT_TRAIN_ENVS_DIR + "/g1/g1_mimic_distill_config.py", policy="now")
    
    env, _ = task_registry.make_env(name=args.task, args=args)
    print(f"Using motion file: {env.cfg.motion.motion_file}")
    ppo_runner, train_cfg = task_registry.make_alg_runner(log_root=log_pth, env=env, name=args.task, args=args)
    ppo_runner.learn(num_learning_iterations=train_cfg.runner.max_iterations, init_at_random_ep_len=True)
    

if __name__ == "__main__":
    args = get_args()
    train(args)
