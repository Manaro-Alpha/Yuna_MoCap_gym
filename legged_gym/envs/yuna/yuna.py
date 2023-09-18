from time import time
import numpy as np
import os

from isaacgym.torch_utils import *
from isaacgym import gymtorch, gymapi, gymutil

import torch
from typing import Tuple, Dict
from legged_gym.envs import LeggedRobot

class Yuna(LeggedRobot):
    # def __init__(self):
        
    
    def _custom_reset(self,env_ids):
        # print("_custom_reset")
        motion_array = np.load('/home/mananaro/Yuna_MoCap_gym/data/Yuna_train_data.npy')
        motion_array_pos = np.zeros(motion_array.shape)
        for i in range(motion_array.shape[0]):
            for j in range(motion_array.shape[1]):
                motion_array_pos[i][j] = motion_array[i][j][0]
        self.reset_index = np.random.randint(0,motion_array.shape[0],self.num_envs)
        self.default_dof_pos_ = torch.zeros((self.num_envs,self.num_dof),dtype=torch.float, device=self.device, requires_grad=False)
        # print("def_pos shape = ",self.dof_pos[env_ids].shape)
        for n in range(self.num_envs):
            for j in range(self.num_dofs):
                # name = self.dof_names[j]
                # self.init_joint_angle_index[n] = self.cfg.init_state.joint_idx
                # angle = self.cfg.init_state.default_joint_angles[name]
                angle = motion_array_pos[self.reset_index[n]][j]
                self.default_dof_pos_[n][j] = angle


                # self.default_dof_pos_[n][j] = angle
        self.dof_pos = self.default_dof_pos_ + torch_rand_float(0.1,0.01,shape =(self.num_envs,self.num_dof),device=self.device)
        # print(self.dof_pos)

        self.dof_vel[env_ids] = 0.

        env_ids_int32 = env_ids.to(dtype=torch.int32)
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self.dof_state),
                                              gymtorch.unwrap_tensor(env_ids_int32), len(env_ids_int32))


    def _reward_no_fly(self):
        contacts = self.contact_forces[:, self.feet_indices, 2] > 0.1
        single_contact = torch.sum(1.*contacts, dim=1)==3
        return 1.*single_contact
    
    def _reward_pose_match(self):
        # self.init_joint_angle_index = torch.arange(8)
        # self.init_joint_angle_index = self.init_joint_angle_index.view(self.num_envs,8)
        motion_array = np.load('/home/mananaro/Yuna_MoCap_gym/data/Yuna_train_data.npy')
        motion_array_pos = np.zeros(motion_array.shape)
        for i in range(motion_array.shape[0]):
            for j in range(motion_array.shape[1]):
                motion_array_pos[i][j] = motion_array[i][j][0]
        # joint_angle_array = joint_angle_array + noise
        motion_tensor_pos = torch.Tensor(motion_array_pos).to(self.device)
        current_frame_idx = self.reset_index
        goal_frame_idx = current_frame_idx + 1
        goal_frame_idx = goal_frame_idx%motion_array.shape[0]
        # print(goal_frame_idx,current_frame_idx)
        # dof_pos = joint_angle_tensor
        pos_dist = 0
        for i in range(self.num_envs):
            pos_dist += torch.exp(-2*torch.norm(self.dof_pos[i]-motion_tensor_pos[goal_frame_idx[i]],p=2))
        # pos_dist = pos_dist.exp()
        return pos_dist
    
    def _reward_vel_match(self):

        motion_array = np.load('/home/mananaro/Yuna_MoCap_gym/data/Yuna_train_data.npy')
        motion_array_vel = np.zeros(motion_array.shape)
        for i in range(motion_array.shape[0]):
            for j in range(motion_array.shape[1]):
                motion_array_vel[i][j] = motion_array[i][j][1]
        motion_tensor_vel = torch.Tensor(motion_array_vel).to(self.device)
        current_frame_idx = self.reset_index
        goal_frame_idx = current_frame_idx + 1
        goal_frame_idx = goal_frame_idx%motion_array.shape[0]
        vel_dist = 0
        for i in range(self.num_envs):
            vel_dist += torch.exp(-2*torch.norm(self.dof_vel[i]-motion_tensor_vel[goal_frame_idx[i]],p=2))
        # pos_dist = pos_dist.exp()
        return vel_dist



