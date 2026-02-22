# SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-3-Clause
#
# Copyright (c) 2021 ETH Zurich, Nikita Rudin

import os

# TELEOPIT_TRAIN_ROOT_DIR points to teleopit_train/ directory
# (parent of legged_gym/, sibling of assets/, pose/, rsl_rl/)
TELEOPIT_TRAIN_ROOT_DIR = os.path.dirname(os.path.realpath(__file__))
TELEOPIT_TRAIN_ROOT_DIR = os.path.dirname(TELEOPIT_TRAIN_ROOT_DIR)
TELEOPIT_TRAIN_ENVS_DIR = os.path.join(TELEOPIT_TRAIN_ROOT_DIR, 'legged_gym', 'envs')
POSE_DIR = os.path.join(TELEOPIT_TRAIN_ROOT_DIR, 'pose')
