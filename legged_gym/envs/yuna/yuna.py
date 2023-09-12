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
        self.init_joint_angle_index = torch.zeros(self.num_envs,device=self.device,requires_grad=False)
        self.default_dof_pos_ = torch.zeros((self.num_envs,self.num_dof),dtype=torch.float, device=self.device, requires_grad=False)
        # print("def_pos shape = ",self.dof_pos[env_ids].shape)
        for n in range(self.num_envs):
            for j in range(self.num_dofs):
                name = self.dof_names[j]
                self.init_joint_angle_index[n] = self.cfg.init_state.joint_idx
                angle = self.cfg.init_state.default_joint_angles[name]
                self.default_dof_pos_[n][j] = angle


                # self.default_dof_pos_[n][j] = angle
        self.dof_pos = self.default_dof_pos_ + torch_rand_float(0.01,0.01,shape =(self.num_envs,self.num_dof),device=self.device)
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
        joint_angle_array = np.loadtxt('/home/mananaro/Yuna_MoCap_gym/data/Yuna_walk_train_data.txt',dtype=np.float32,delimiter=',')
        noise = np.random.normal(0,0.01,joint_angle_array.shape)
        joint_angle_array = joint_angle_array + noise

        joint_angle_tensor = torch.Tensor(joint_angle_array).to(self.device)
        rewards = torch.zeros((self.num_envs)).to(self.device)
        reward_env = torch.zeros((6)).to(self.device)
        for i in range(self.num_envs):
            for j in range(6):
                reward = torch.norm((self.dof_pos[i] - joint_angle_tensor[j]),p=2)
                reward_env[j] = (reward)
            rewards[i] = torch.argmin(reward_env)

        rewards = rewards + 1
        rewards = rewards%6
        # print(joint_angle_tensor)
        # dof_pos = joint_angle_tensor
        pos_dist = -2*rewards
        post_dist = pos_dist.exp()
        return post_dist.sum(dim=0)


