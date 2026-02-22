from teleopit_train.legged_gym.envs.base.humanoid_mimic_config import HumanoidMimicCfg, HumanoidMimicCfgPPO
from teleopit_train.legged_gym import TELEOPIT_TRAIN_ROOT_DIR


class G1MimicPrivCfg(HumanoidMimicCfg):
    class env(HumanoidMimicCfg.env):
        tar_motion_steps_priv = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        # usually student obs_steps is the subset of priv obs_steps
        tar_motion_steps = [1, 5, 10, 15, 20, 25, 30, 35, 40, 45,
                         50, 55, 60, 65, 70, 75, 80, 85, 90, 95,]
        
        num_envs = 4096
        num_actions = 29
        obs_type = 'priv' # 'student'
        n_priv_latent = 4 + 1 + 2*num_actions
        extra_critic_obs = 3
        n_priv = 0
        
        n_proprio = 3 + 2 + 3*num_actions
        n_priv_mimic_obs = len(tar_motion_steps_priv) * (21 + num_actions + 3*9) # Hardcode for now, 9 is base, 9 is the number of key bodies
        n_mimic_obs_single = 6 + 29  # Modified: root_vel_xy(2) + root_pos_z(1) + roll_pitch(2) + yaw_ang_vel(1) + dof_pos(29)
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_priv_info = 3 + 3 + 4 + 3*9 + 2 + 4 + 1 + 2*num_actions # base lin vel, root height, key body pos, contact mask, priv latent
        history_len = 10
        
        n_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        n_priv_obs_single = n_priv_mimic_obs + n_proprio + n_priv_info
        
        num_observations = n_priv_obs_single

        num_privileged_obs = n_priv_obs_single

        env_spacing = 3.  # not used with heightfields/trimeshes 
        send_timeouts = True # send time out information to the algorithm
        episode_length_s = 10
        
        randomize_start_pos = True
        randomize_start_yaw = False
        
        history_encoding = True
        contact_buf_len = 10
        
        normalize_obs = True
        
        enable_early_termination = True
        pose_termination = True
        pose_termination_dist = 0.7
        rand_reset = True
        
        track_root = False
        root_tracking_termination_dist = 2.0
     
     
        dof_err_w = [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Left Leg
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Right Leg
                     1.0, 1.0, 1.0, # waist yaw, roll, pitch
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Left Arm
                     1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, # Right Arm
                     ]
        
        
        global_obs = False
        # global_obs = True
    
    class terrain(HumanoidMimicCfg.terrain):
        # mesh_type = 'trimesh'
        mesh_type = 'plane'
        # height = [0, 0.02]
        height = [0, 0.00]
        horizontal_scale = 0.1
    
    class init_state(HumanoidMimicCfg.init_state):
        pos = [0, 0, 1.0]
        default_joint_angles = {
            'left_hip_pitch_joint': -0.2,
            'left_hip_roll_joint': 0.0,
            'left_hip_yaw_joint': 0.0,
            'left_knee_joint': 0.4,
            'left_ankle_pitch_joint': -0.2,
            'left_ankle_roll_joint': 0.0,
            
            'right_hip_pitch_joint': -0.2,
            'right_hip_roll_joint': 0.0,
            'right_hip_yaw_joint': 0.0,
            'right_knee_joint': 0.4,
            'right_ankle_pitch_joint': -0.2,
            'right_ankle_roll_joint': 0.0,
            
            'waist_yaw_joint': 0.0,
            'waist_roll_joint': 0.0,
            'waist_pitch_joint': 0.0,
            
            'left_shoulder_pitch_joint': 0.0,
            'left_shoulder_roll_joint': 0.4,
            'left_shoulder_yaw_joint': 0.0,
            'left_elbow_joint': 1.2,
            'left_wrist_roll_joint': 0.0,
            'left_wrist_pitch_joint': 0.0,
            'left_wrist_yaw_joint': 0.0,
            
            'right_shoulder_pitch_joint': 0.0,
            'right_shoulder_roll_joint': -0.4,
            'right_shoulder_yaw_joint': 0.0,
            'right_elbow_joint': 1.2,
            'right_wrist_roll_joint': 0.0,
            'right_wrist_pitch_joint': 0.0,
            'right_wrist_yaw_joint': 0.0,
        }
    
    class control(HumanoidMimicCfg.control):
                
        stiffness = {'hip_yaw': 100,
                    'hip_roll': 100,
                    'hip_pitch': 100,
                    'knee': 150,
                    'ankle': 40,
                    'waist': 150,
                    'shoulder': 40,
                    'elbow': 40,
                    'wrist': 40,
                    }  # [N*m/rad]
        damping = {  'hip_yaw': 2,
                    'hip_roll': 2,
                    'hip_pitch': 2,
                    'knee': 4,
                    'ankle': 2,
                    'waist': 4,
                    'shoulder': 5,
                    'elbow': 5,
                    'wrist': 5,
                    }  # [N*m/rad]  # [N*m*s/rad]

        action_scale = 0.5
        decimation = 10
        # decimation = 4
    
    class sim(HumanoidMimicCfg.sim):
        dt = 0.002 # 1/500
        # dt = 1/200 # 0.005
        
    class normalization(HumanoidMimicCfg.normalization):
        clip_actions = 5.0
    
    class asset(HumanoidMimicCfg.asset):
        # file = f'{TELEOPIT_TRAIN_ROOT_DIR}/assets/g1/g1_custom_collision.urdf'
        # file = f'{TELEOPIT_TRAIN_ROOT_DIR}/assets/g1/g1_custom_collision_with_fixed_hand.urdf'
        file = f'{TELEOPIT_TRAIN_ROOT_DIR}/assets/g1/g1_custom_collision_29dof.urdf'
        
        # for both joint and link name
        torso_name: str = 'pelvis'  # humanoid pelvis part
        chest_name: str = 'imu_in_torso'  # humanoid chest part

        # for link name
        thigh_name: str = 'hip'
        shank_name: str = 'knee'
        foot_name: str = 'ankle_roll_link'  # foot_pitch is not used
        waist_name: list = ['torso_link', 'waist_roll_link', 'waist_yaw_link']
        upper_arm_name: str = 'shoulder_roll_link'
        lower_arm_name: str = 'elbow_link'
        hand_name: list = ['right_rubber_hand', 'left_rubber_hand']

        feet_bodies = ['left_ankle_roll_link', 'right_ankle_roll_link']
        n_lower_body_dofs: int = 12

        penalize_contacts_on = ["shoulder", "elbow", "hip", "knee"]
        terminate_after_contacts_on = []
        
        
        # ========================= Inertia =========================
        # shoulder, elbow, and ankle: 0.139 * 1e-4 * 16**2 + 0.017 * 1e-4 * (46/18 + 1)**2 + 0.169 * 1e-4 = 0.003597
        # waist, hip pitch & yaw: 0.489 * 1e-4 * 14.3**2 + 0.098 * 1e-4 * 4.5**2 + 0.533 * 1e-4 = 0.0103
        # knee, hip roll: 0.489 * 1e-4 * 22.5**2 + 0.109 * 1e-4 * 4.5**2 + 0.738 * 1e-4 = 0.0251
        # wrist: 0.068 * 1e-4 * 25**2 = 0.00425
        
        # dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 3 + [0.003597] * 8
        dof_armature = [0.0103, 0.0251, 0.0103, 0.0251, 0.003597, 0.003597] * 2 + [0.0103] * 3 + \
            [0.003597, 0.003597, 0.003597, 0.003597, 0.003597, 0.00425, 0.00425] * 2
        
        
        # ========================= Inertia =========================
        
        collapse_fixed_joints = False
    
    class rewards(HumanoidMimicCfg.rewards):
        regularization_names = [
                        # "feet_stumble",
                        # "feet_contact_forces",
                        # "lin_vel_z",
                        # "ang_vel_xy",
                        # "orientation",
                        # "dof_pos_limits",
                        # "dof_torque_limits",
                        # "collision",
                        # "torque_penalty",
                        # "thigh_torque_roll_yaw",
                        # "thigh_roll_yaw_acc",
                        # "dof_acc",
                        # "dof_vel",
                        # "action_rate",
                        ]
        regularization_scale = 1.0
        regularization_scale_range = [0.8,2.0]
        regularization_scale_curriculum = False
        regularization_scale_gamma = 0.0001
        
        class scales:      
            # 0628 version  
            # tracking_joint_dof = 0.6
            tracking_joint_dof = 2.0
            tracking_joint_vel = 0.2
            # tracking_root_translation_xy = 1.0
            tracking_root_translation_z = 1.0
            tracking_root_rotation = 1.0
            tracking_root_linear_vel = 1.0
            tracking_root_angular_vel = 1.0
            tracking_keybody_pos = 2.0
            tracking_keybody_pos_global = 2.0
            alive = 0.5
            feet_slip = -0.1
            feet_contact_forces = -5e-4      
            feet_stumble = -1.25
            dof_pos_limits = -5.0
            dof_torque_limits = -1.0
            dof_vel = -1e-4
            dof_acc = -5e-8
            action_rate = -0.01
            feet_air_time = 5.0
            ang_vel_xy = -0.01            
            ankle_dof_acc = -5e-8 * 2
            ankle_dof_vel = -1e-4 * 2
            
            
            # TWIST version
            # tracking_joint_dof = 0.6
            # tracking_joint_vel = 0.2
            # tracking_root_pose = 0.6
            # tracking_root_vel = 1.0
            # tracking_keybody_pos = 2.0
            # feet_slip = -0.1
            # feet_contact_forces = -5e-4      
            # feet_stumble = -1.25
            # dof_pos_limits = -5.0
            # dof_torque_limits = -1.0
            # dof_vel = -1e-4
            # dof_acc = -5e-8
            # action_rate = -0.01
            # feet_air_time = 5.0
            # ang_vel_xy = -0.01            
            # ankle_dof_acc = -5e-8 * 2
            # ankle_dof_vel = -1e-4 * 2
            

        min_dist = 0.1
        max_dist = 0.4
        max_knee_dist = 0.4
        feet_height_target = 0.2
        feet_air_time_target = 0.5
        only_positive_rewards = False
        tracking_sigma = 0.2
        tracking_sigma_ang = 0.125
        max_contact_force = 500  # Forces above this value are penalized
        soft_torque_limit = 0.95
        torque_safety_limit = 0.9
        
        # =========================
        termination_roll = 4.0
        termination_pitch = 4.0
        root_height_diff_threshold = 0.3
        

    class evaluations:
        tracking_joint_dof = True
        tracking_joint_vel = True
        tracking_root_translation = True
        tracking_root_rotation = True
        tracking_root_vel = True
        tracking_root_ang_vel = True
        tracking_keybody_pos = True
        tracking_root_pose_delta_local = True
        tracking_root_rotation_delta_local = True
        
        
    class domain_rand:
        domain_rand_general = True # manually set this, setting from parser does not work;
        
        randomize_gravity = (True and domain_rand_general)
        gravity_rand_interval_s = 4
        gravity_range = (-0.1, 0.1)
        
        randomize_friction = (True and domain_rand_general)
        friction_range = [0.1, 2.]
        
        randomize_base_mass = (True and domain_rand_general)
        added_mass_range = [-3., 3]
        
        randomize_base_com = (True and domain_rand_general)
        added_com_range = [-0.05, 0.05]
        
        push_robots = (True and domain_rand_general)
        push_interval_s = 4
        max_push_vel_xy = 1.0
        
        push_end_effector = (False and domain_rand_general)
        push_end_effector_interval_s = 2
        max_push_force_end_effector = 10.0

        randomize_motor = (True and domain_rand_general)
        motor_strength_range = [0.8, 1.2]

        action_delay = (True and domain_rand_general)
        action_buf_len = 8
    
    class noise(HumanoidMimicCfg.noise):
        # add_noise = False
        add_noise = True
        noise_increasing_steps = 50_000
        class noise_scales:
            dof_pos = 0.01
            dof_vel = 0.1
            lin_vel = 0.1
            ang_vel = 0.1
            gravity = 0.05
            imu = 0.1
        
    class motion(HumanoidMimicCfg.motion):
        motion_curriculum = True
        motion_curriculum_gamma = 0.01
        reset_consec_frames = 30
        key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link", "left_elbow_link", "right_elbow_link", "head_mocap"] # 9 key bodies
        upper_key_bodies = ["left_rubber_hand", "right_rubber_hand", "left_elbow_link", "right_elbow_link", "head_mocap"]
        sample_ratio = 1.0
        motion_smooth = True
        motion_decompose = False

        motion_file = f"{TELEOPIT_TRAIN_ROOT_DIR}/motion_data_configs/g1_omomo+mocap_static+amass_walk.yaml"
        


class G1MimicStuCfg(G1MimicPrivCfg):
    class env(G1MimicPrivCfg.env):
        obs_type = 'student'
        tar_motion_steps = [1]
        n_mimic_obs_single = G1MimicPrivCfg.env.n_mimic_obs_single
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = G1MimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (G1MimicPrivCfg.env.history_len + 1)


class G1MimicStuRLCfg(G1MimicPrivCfg):
    class env(G1MimicPrivCfg.env):
        obs_type = 'student'
        tar_motion_steps = [1]
        n_mimic_obs_single = G1MimicPrivCfg.env.n_mimic_obs_single
        n_mimic_obs = len(tar_motion_steps) * n_mimic_obs_single
        n_proprio = G1MimicPrivCfg.env.n_proprio
        n_obs_single = n_mimic_obs + n_proprio
        num_observations = n_obs_single * (G1MimicPrivCfg.env.history_len + 1)



class G1MimicPrivCfgPPO(HumanoidMimicCfgPPO):
    seed = 1
    class runner(HumanoidMimicCfgPPO.runner):
        policy_class_name = 'ActorCriticMimic'
        algorithm_class_name = 'PPO'
        runner_class_name = 'OnPolicyRunnerMimic'
        max_iterations = 1_000_002 # number of policy updates

        # logging
        save_interval = 500 # check for potential saves every this many iterations
        experiment_name = 'test'
        run_name = ''
        # load and resume
        resume = False
        load_run = -1 # -1 = last run
        checkpoint = -1 # -1 = last saved model
        resume_path = None # updated from load_run and chkpt
    
    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        # Transformer params
        # learning_rate = 1e-4 #1.e-3 #5.e-4
        # schedule = 'fixed' # could be adaptive, fixed
    
    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        
        
class G1MimicStuCfgDAgger(G1MimicPrivCfgPPO):
    seed = 1
    
    class teachercfg(G1MimicPrivCfgPPO):
        pass
    
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'DAggerActor'
        algorithm_class_name = 'DAgger'
        runner_class_name = 'DAggerRunner'
        max_iterations = 1_000_002
        warm_iters = 100
        
        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'g1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm:
        num_learning_epochs = 5
        num_mini_batches = 4
        learning_rate = 1e-4 #1.e-3 #5.e-4
        max_grad_norm = 1.0
        normalizer_update_iterations = 1000

    class policy:
        actor_hidden_dims = [1024, 1024, 512, 256] # best now
        
        # actor_hidden_dims = [512, 512, 256, 128] # worse than above
        
        history_latent_dim = 128
        activation = 'silu' # can be elu, relu, selu, crelu, lrelu, tanh, sigmoid
        

class G1MimicStuRLCfgDAgger(G1MimicStuRLCfg):
    seed = 1
    
    class teachercfg(G1MimicPrivCfgPPO):
        pass
    
    class runner(G1MimicPrivCfgPPO.runner):
        policy_class_name = 'ActorCriticTeleop'
        algorithm_class_name = 'DaggerPPO'
        runner_class_name = 'OnPolicyDaggerRunner'
        max_iterations = 1_000_002
        warm_iters = 100
        
        # logging
        save_interval = 500
        experiment_name = 'test'
        run_name = ''
        resume = False
        load_run = -1
        checkpoint = -1
        resume_path = None
        
        teacher_experiment_name = 'test'
        teacher_proj_name = 'g1_priv_mimic'
        teacher_checkpoint = -1
        eval_student = False

    class algorithm(HumanoidMimicCfgPPO.algorithm):
        grad_penalty_coef_schedule = [0.00, 0.00, 700, 1000]
        std_schedule = [1.0, 0.4, 4000, 1500]
        entropy_coef = 0.005
        
        dagger_coef_anneal_steps = 60000  # Total steps to anneal dagger_coef to dagger_coef_min
        
        # dagger_coef = 0.1
        # dagger_coef_min = 0.01  # Minimum value for dagger_coef
        dagger_coef = 0.2
        dagger_coef_min = 0.1
        # dagger_coef = 0.0
        # dagger_coef_min = 0.0  # Minimum value for dagger_coef

    class policy(HumanoidMimicCfgPPO.policy):
        action_std = [0.7] * 12 + [0.4] * 3 + [0.5] * 14
        init_noise_std = 1.0
        obs_context_len = 11
        actor_hidden_dims = [512, 512, 256, 128]
        critic_hidden_dims = [512, 512, 256, 128]
        activation = 'silu'
        layer_norm = True
        motion_latent_dim = 128
        


   