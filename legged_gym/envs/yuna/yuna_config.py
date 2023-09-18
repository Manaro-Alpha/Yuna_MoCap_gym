import torch
import numpy as np
from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg,LeggedRobotCfgPPO

class YunaCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 64
        num_observations = 66
        num_actions = 18
    
    class terrain(LeggedRobotCfg.terrain):
        curriculum = False
        mesh_type = 'plane'
        measure_heights = False
    
    class init_state(LeggedRobotCfg.init_state):
        joint_angle_array = np.loadtxt('/home/mananaro/Yuna_MoCap_gym/data/Yuna_walk_train_data.txt',dtype=np.float32,delimiter=',')
        noise = np.random.normal(0,0.001,joint_angle_array.shape)
        joint_angle_array = joint_angle_array + noise
        pos = [0.0,0.0,0.3]
        joint_idx = np.random.randint(0,joint_angle_array.shape[0])
        # default_joint_angles = { # = target angles [rad] when action = 0.0
        #     'base1': 0.,
        #     'base2': 0.,
        #     'base3': 0.,
        #     'base4': 0.,
        #     'base5': 0.,
        #     'base6': 0.,

        #     'shoulder1': 0.,
        #     'shoulder2': 0.,
        #     'shoulder3': 0.,
        #     'shoulder4': 0.,
        #     'shoulder5': 0.,
        #     'shoulder6': 0.,

        #     'elbow1': -1.5708 + 0.05,    # - pi/2
        #     'elbow2': 1.5708 - 0.05,     #   pi/2
        #     'elbow3': -1. + 0.05,    # - pi/2
        #     'elbow4': 1.5708 - 0.05,     #   pi/2
        #     'elbow5': -1.5708 + 0.05,    # - pi/2
        #     'elbow6': 1.5708 - 0.05,     #   pi/2
        # }
        default_joint_angles = {
                'base1': joint_angle_array[joint_idx][0],
                'base2': joint_angle_array[joint_idx][3],
                'base3': joint_angle_array[joint_idx][6],
                'base4': joint_angle_array[joint_idx][9],
                'base5': joint_angle_array[joint_idx][12],
                'base6': joint_angle_array[joint_idx][15],

                'shoulder1': joint_angle_array[joint_idx][1],
                'shoulder2': joint_angle_array[joint_idx][4],
                'shoulder3': joint_angle_array[joint_idx][7],
                'shoulder4': joint_angle_array[joint_idx][10],
                'shoulder5': joint_angle_array[joint_idx][13],
                'shoulder6': joint_angle_array[joint_idx][16],

                'elbow1': joint_angle_array[joint_idx][2],
                'elbow2': joint_angle_array[joint_idx][5],
                'elbow3': joint_angle_array[joint_idx][8],
                'elbow4': joint_angle_array[joint_idx][11],
                'elbow5': joint_angle_array[joint_idx][14],
                'elbow6': joint_angle_array[joint_idx][17],
}
        # def get_joint_angles(self,i):
        #     joint_angle_array = np.loadtxt('data/Yuna_walk_train_data.txt',dtype=np.float32,delimiter=',')
        #     noise = np.random.normal(0,0.01,joint_angle_array.shape)
        #     joint_angle_array = joint_angle_array + noise
        #     default_joint_angles = {
        #         'base1': joint_angle_array[i][0],
        #         'base2': joint_angle_array[i][3],
        #         'base3': joint_angle_array[i][6],
        #         'base4': joint_angle_array[i][9],
        #         'base5': joint_angle_array[i][12],
        #         'base6': joint_angle_array[i][15],

        #         'shoulder1': joint_angle_array[i][1],
        #         'shoulder2': joint_angle_array[i][4],
        #         'shoulder3': joint_angle_array[i][7],
        #         'shoulder4': joint_angle_array[i][10],
        #         'shoulder5': joint_angle_array[i][13],
        #         'shoulder6': joint_angle_array[i][16],

        #         'elbow1': joint_angle_array[i][2],
        #         'elbow2': joint_angle_array[i][5],
        #         'elbow3': joint_angle_array[i][8],
        #         'elbow4': joint_angle_array[i][11],
        #         'elbow5': joint_angle_array[i][14],
        #         'elbow6': joint_angle_array[i][17],
        #     }
        #     return default_joint_angles

    class control(LeggedRobotCfg.control):
        stiffness = {'base1': 1000.,'base2': 1000.,'base3': 1000.,'base4': 1000.,'base5': 1000.,'base6': 1000.,
                     'shoulder1': 1000.,'shoulder2': 1000.,'shoulder3': 1000.,'shoulder4': 1000.,'shoulder5': 1000.,'shoulder6': 1000.,
                     'elbow1': 1000.,'elbow2': 1000.,'elbow3': 1000.,'elbow4': 1000.,'elbow5': 1000.,'elbow6': 1000.}
        
        damping = {'base1': 100,'base2': 100,'base3': 100,'base4': 100,'base5': 100,'base6': 100,
                     'shoulder1': 100,'shoulder2': 100,'shoulder3': 100,'shoulder4': 100,'shoulder5': 100,'shoulder6': 100,
                     'elbow1': 100,'elbow2': 100,'elbow3': 100,'elbow4': 100,'elbow5': 100,'elbow6': 100}
        
        action_scale = 0.25
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/yuna/urdf/yuna.urdf'
        foot_name = 'FOOT'
        penalize_contacts_on = {'LINK'}
        terminate_after_contacts_on = {'base_link'}
        flip_visual_attachments = False
        self_collisions = 1
    
    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False
        class scales( LeggedRobotCfg.rewards.scales ):
            termination = -200.
            tracking_ang_vel = 0.5
            torques = -5.e-4
            dof_acc = -2.e-7
            lin_vel_z = -0.5
            feet_air_time = 5.
            dof_pos_limits = -1.
            no_fly = 0.25
            dof_vel = -0.0
            ang_vel_xy = -0.0
            feet_contact_forces = -0.
            pose_match = 0.65
            vel_match = 0.1

class YunaCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = 'yuna_test'
        experiment_name = 'flat'