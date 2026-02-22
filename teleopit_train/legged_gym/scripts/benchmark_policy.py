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

from teleopit_train.legged_gym.envs import *
from teleopit_train.legged_gym.gym_utils import get_args, task_registry
import torch
import faulthandler
from tqdm import tqdm
from termcolor import cprint
import numpy as np
def get_load_path(root, load_run=-1, checkpoint=-1, model_name_include="jit"):
    if checkpoint==-1:
        models = [file for file in os.listdir(root) if model_name_include in file]
        models.sort(key=lambda m: '{0:0>15}'.format(m))
        model = models[-1]
        checkpoint = model.split("_")[-1].split(".")[0]
    return model, checkpoint

def set_play_cfg(env_cfg):
    env_cfg.env.num_envs = 1#2 if not args.num_envs else args.num_envs
    env_cfg.env.episode_length_s = 60
    # env_cfg.commands.resampling_time = 60
    env_cfg.terrain.num_rows = 5
    env_cfg.terrain.num_cols = 5
    env_cfg.terrain.curriculum = False
    env_cfg.terrain.max_difficulty = True
    
    env_cfg.noise.add_noise = False
    env_cfg.domain_rand.randomize_friction = True
    env_cfg.domain_rand.push_robots = False
    env_cfg.domain_rand.push_interval_s = 5
    env_cfg.domain_rand.max_push_vel_xy = 2.5
    env_cfg.domain_rand.randomize_base_mass = False
    env_cfg.domain_rand.randomize_base_com = False
    env_cfg.domain_rand.action_delay = False
    
    if hasattr(env_cfg, "motion"):
        env_cfg.motion.motion_curriculum = False


def play(args):
    faulthandler.enable()
    
    exptid = args.exptid
    log_pth = "../../logs/{}/".format(args.proj_name) + args.exptid

    env_cfg, train_cfg = task_registry.get_cfgs(name=args.task)

    set_play_cfg(env_cfg)

    env_cfg.env.record_video = args.record_video
    if_normalize = env_cfg.env.normalize_obs
    cprint(f"if_normalize: {if_normalize}", "green")
    if env_cfg.env.record_video:
        env_cfg.env.episode_length_s = 10

    # change motion file
    # env_cfg.motion.motion_file = "/home/yanjieze/projects/g1_wbc/humanoid-motion-imitation/assets/mocap_data/converted/BoxLift2.pkl"
    # env_cfg.motion.motion_file = f"{TELEOPIT_TRAIN_ROOT_DIR}/motion_data_configs/g1_mocap_test.yaml"
    env_cfg.motion.motion_file = f"{TELEOPIT_TRAIN_ROOT_DIR}/motion_data_configs/g1_mocap_origin_test.yaml"
    env_cfg.env.rand_reset = False
    env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    obs = env.get_observations()

    # load policy
    train_cfg.runner.resume = True
    ppo_runner, train_cfg, log_pth = task_registry.make_alg_runner(log_root = log_pth, env=env, name=args.task, args=args, train_cfg=train_cfg, return_log_dir=True)

    if args.use_jit:
        path = os.path.join(log_pth, "traced")
        model, checkpoint = get_load_path(root=path, checkpoint=args.checkpoint)
        path = os.path.join(path, model)
        print("Loading jit for policy: ", path)
        policy_jit = torch.jit.load(path, map_location=env.device)
        print("policy_jit: ", policy_jit)
    else:
        policy = ppo_runner.get_inference_policy(device=env.device)
        if if_normalize:
            try:
                normalizer = ppo_runner.get_normalizer(device=env.device)
            except:
                print("No normalizer found")
                normalizer = None
        print("policy: ", policy)

    actions = torch.zeros(env.num_envs, env.num_actions, device=env.device, requires_grad=False)

    if args.record_video:
        mp4_writers = []
        import imageio
        env.enable_viewer_sync = True
        # env.enable_viewer_sync = False
        for i in range(env.num_envs):
            video_name = args.proj_name + "-" + args.exptid +".mp4"
            run_name = log_pth.split("/")[-1]
            path = f"../../logs/videos_retarget/{run_name}"
            if not os.path.exists(path):
                os.makedirs(path)
            video_name = os.path.join(path, video_name)
            mp4_writer = imageio.get_writer(video_name, fps=int(1/env.dt))
            cprint(f"Recording video to {video_name}", "green")
            mp4_writers.append(mp4_writer)

    if args.record_log:
        import json
        run_name = log_pth.split("/")[-1]
        logs_dict = []
        dict_name = args.proj_name + "-" + args.exptid + ".json"
        path = f"../../logs/env_logs/{run_name}"
        if not os.path.exists(path):
            os.makedirs(path)
        dict_name = os.path.join(path, dict_name)
        

    env_id = env.lookat_id

    num_motions = env._motion_lib.num_motions()
    cprint(f"num_motions: {num_motions}", "green")
    
    # env, _ = task_registry.make_env(name=args.task, args=args, env_cfg=env_cfg)
    error_tracking_joint_dof = []
    error_tracking_joint_vel = []
    error_tracking_root_translation = []
    error_tracking_root_rotation = []
    error_tracking_root_vel = []
    error_tracking_root_ang_vel = []
    error_tracking_keybody_pos = []
    error_feet_slip = []
    
    # error tracking keybody pos
    error_tracking_keybody_pos_hand = []
    error_tracking_keybody_pos_feet = []
    error_tracking_keybody_pos_knee = []
    error_tracking_keybody_pos_elbow = []
    error_tracking_keybody_pos_head = []
    
    for motion_id in tqdm(range(num_motions)):
        motion_id = torch.tensor([motion_id], device=env.device, dtype=torch.long)
        motion_time = torch.zeros((1,), device=env.device, dtype=torch.float)
        motion_length = env._motion_lib.get_motion_length(motion_id)
        # traj_length = int(env.max_episode_length) // 10
        # traj_length = int(env.max_episode_length) 
        traj_length = int(motion_length / env.dt) * 1
        reset_env_ids = torch.tensor([0], device=env.device, dtype=torch.long)
        env.reset_idx(env_ids=reset_env_ids, motion_ids=motion_id)
        
        
        for t in range(traj_length):
            
            env.gym.simulate(env.sim)
            env.gym.fetch_results(env.sim, True)
            env.gym.refresh_rigid_body_state_tensor(env.sim)
        
            obs = env.get_observations()
        
            if args.use_jit:
                actions = policy_jit(obs.detach())
            else:
                if if_normalize and normalizer is not None:
                    normalized_obs = normalizer.normalize(obs.detach())
                else:
                    normalized_obs = obs.detach()
                actions = policy(normalized_obs, hist_encoding=True)
                
            obs, _, rews, dones, infos = env.step(actions.detach())
            
            error_tracking_joint_dof.append(env._error_tracking_joint_dof().item())
            error_tracking_joint_vel.append(env._error_tracking_joint_vel().item())
            error_tracking_root_translation.append(env._error_tracking_root_translation().item())
            error_tracking_root_rotation.append(env._error_tracking_root_rotation().item())
            error_tracking_root_vel.append(env._error_tracking_root_vel().item())
            error_tracking_root_ang_vel.append(env._error_tracking_root_ang_vel().item())
            error_tracking_keybody_pos_single, error_tracking_keybody_pos_diff = env._error_tracking_keybody_pos()
            
            # key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link", "head_mocap"] # 9 key bodies
            keybody_hand_err = (error_tracking_keybody_pos_diff[0,0]+error_tracking_keybody_pos_diff[0,1])/2
            error_tracking_keybody_pos_hand.append(keybody_hand_err.item())
            keybody_feet_err = (error_tracking_keybody_pos_diff[0,2]+error_tracking_keybody_pos_diff[0,3])/2
            error_tracking_keybody_pos_feet.append(keybody_feet_err.item())
            keybody_knee_err = (error_tracking_keybody_pos_diff[0,4]+error_tracking_keybody_pos_diff[0,5])/2
            error_tracking_keybody_pos_knee.append(keybody_knee_err.item())
            keybody_elbow_err = (error_tracking_keybody_pos_diff[0,6]+error_tracking_keybody_pos_diff[0,7])/2
            error_tracking_keybody_pos_elbow.append(keybody_elbow_err.item())
            keybody_head_err = error_tracking_keybody_pos_diff[0,8]
            error_tracking_keybody_pos_head.append(keybody_head_err.item())
            
            error_tracking_keybody_pos.append(error_tracking_keybody_pos_single.item())
            error_feet_slip.append(env._error_feet_slip().item())
            
            if args.record_video:
                imgs = env.render_record(mode='rgb_array')
                if imgs is not None:
                    for i in range(env.num_envs):
                        mp4_writers[i].append_data(imgs[i])
                        
            if args.record_log:
                log_dict = env.get_episode_log()
                logs_dict.append(log_dict)
            
            # Interaction
            if env.button_pressed:
                print(f"env_id: {env.lookat_id:<{5}}")
    
    total_error = np.mean(error_tracking_joint_dof) + np.mean(error_tracking_joint_vel) + np.mean(error_tracking_root_translation) + np.mean(error_tracking_root_rotation) + np.mean(error_tracking_root_vel) + np.mean(error_tracking_keybody_pos) + np.mean(error_feet_slip)
    total_error += np.mean(error_tracking_root_ang_vel)
    
    # print avg error
    cprint(f"Policy: {args.exptid}", "green")
    cprint(f"avg error_tracking_joint_dof: {np.mean(error_tracking_joint_dof):.4f}", "green")
    cprint(f"avg error_tracking_joint_vel: {np.mean(error_tracking_joint_vel):.4f}", "green")
    cprint(f"avg error_tracking_root_translation: {np.mean(error_tracking_root_translation):.4f}", "green")
    cprint(f"avg error_tracking_root_rotation: {np.mean(error_tracking_root_rotation):.4f}", "green")
    cprint(f"avg error_tracking_root_vel: {np.mean(error_tracking_root_vel):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos: {np.mean(error_tracking_keybody_pos):.4f}", "green")
    cprint(f"avg error_feet_slip: {np.mean(error_feet_slip):.4f}", "green")
    cprint(f"avg error_tracking_root_ang_vel: {np.mean(error_tracking_root_ang_vel):.4f}", "green")
    
    cprint(f"avg error_tracking_keybody_pos_hand: {np.mean(error_tracking_keybody_pos_hand):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos_feet: {np.mean(error_tracking_keybody_pos_feet):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos_knee: {np.mean(error_tracking_keybody_pos_knee):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos_elbow: {np.mean(error_tracking_keybody_pos_elbow):.4f}", "green")
    cprint(f"avg error_tracking_keybody_pos_head: {np.mean(error_tracking_keybody_pos_head):.4f}", "green")
    
    cprint(f"total_error: {total_error:.4f}", "green")
    
    # output as a txt file
    os.makedirs("benchmark_results", exist_ok=True)
    with open(f"benchmark_results/{args.proj_name}-{args.exptid}-{args.checkpoint}.txt", "w") as f:
        f.write(f"total_error: {total_error:.4f}\n")
        f.write(f"avg error_tracking_joint_dof: {np.mean(error_tracking_joint_dof):.4f}\n")
        f.write(f"avg error_tracking_joint_vel: {np.mean(error_tracking_joint_vel):.4f}\n")
        f.write(f"avg error_tracking_root_translation: {np.mean(error_tracking_root_translation):.4f}\n")
        f.write(f"avg error_tracking_root_rotation: {np.mean(error_tracking_root_rotation):.4f}\n")
        f.write(f"avg error_tracking_root_vel: {np.mean(error_tracking_root_vel):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos: {np.mean(error_tracking_keybody_pos):.4f}\n")
        f.write(f"avg error_feet_slip: {np.mean(error_feet_slip):.4f}\n")
        f.write(f"avg error_tracking_root_ang_vel: {np.mean(error_tracking_root_ang_vel):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_hand: {np.mean(error_tracking_keybody_pos_hand):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_feet: {np.mean(error_tracking_keybody_pos_feet):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_knee: {np.mean(error_tracking_keybody_pos_knee):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_elbow: {np.mean(error_tracking_keybody_pos_elbow):.4f}\n")
        f.write(f"avg error_tracking_keybody_pos_head: {np.mean(error_tracking_keybody_pos_head):.4f}\n")
        print(f"output to benchmark_results/{args.proj_name}-{args.exptid}-{args.checkpoint}.txt")
        
    if args.record_video:
        for mp4_writer in mp4_writers:
            mp4_writer.close()
            
    if args.record_log:
        with open(dict_name, 'w') as f:
            json.dump(logs_dict, f)
    

if __name__ == '__main__':
    args = get_args()
    play(args)