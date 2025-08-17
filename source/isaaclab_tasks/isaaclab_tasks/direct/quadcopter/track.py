# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation
from isaaclab.envs import DirectRLEnv
from isaaclab.markers import VisualizationMarkers
from isaaclab.utils.math import subtract_frame_transforms
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
##
# Pre-defined configs
##
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from isaaclab_tasks.direct.quadcopter.modules.controller import RateController
from isaaclab_tasks.direct.quadcopter.modules.motor import MotorModel

from .quadcopter_cfg import TrackEnvCfg
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate

import random


class TrackEnv(DirectRLEnv):
    cfg: TrackEnvCfg

    def __init__(self, cfg: TrackEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)

        # Total thrust and moment applied to the base of the quadcopter
        self._actions = torch.zeros(self.num_envs, gym.spaces.flatdim(self.single_action_space), device=self.device)
        self._thrust = torch.zeros(self.num_envs, 1, 3, device=self.device)
        self._moment = torch.zeros(self.num_envs, 1, 3, device=self.device)
        # Goal position
        self._desired_pos_w = torch.zeros(self.num_envs, 3, device=self.device)
        
        self.dt = self.cfg.sim.dt
        self.motor_model = MotorModel(self.num_envs, self.device, self.dt, self.cfg.domain_randomization.motor)
        self.rate_controller = RateController(self.num_envs, self.device)
        
        self.last_action = torch.zeros(self.num_envs, 4, device=self.device)
        # self.step_dt=self.dt*self.cfg.decimation
        # Logging
        # 更新 Logging - 添加 yaw_alignment
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "speed_reward",
                "distance_to_goal",
                "action_diff",
                "reward_hover",
            ]
        }
        # Get specific body indices
        self._body_id = self._robot.find_bodies("base_link")[0]
        rotor_order = ["link2", "link1", "link3", "link4"]
        joint_order = ["joint2", "joint1", "joint3", "joint4"]
                
        # 按指定顺序查找 rotors 和 joints
        self._rotors_id = self._robot.find_bodies(rotor_order, preserve_order=True)[0]
        self._rotor_joint_ids = self._robot.find_joints(joint_order, preserve_order=True)[0]
                
        self._robot_mass = self._robot.root_physx_view.get_masses()[0].sum()
        self._gravity_magnitude = torch.tensor(self.sim.cfg.gravity, device=self.device).norm()
        self._robot_weight = (self._robot_mass * self._gravity_magnitude).item()

        self._desired_yaw = torch.zeros(self.num_envs, device=self.device)  # Desired yaw angle for each environment
        self._yaw = torch.zeros(self.num_envs, device=self.device)  # Current yaw angle for each environment
        
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        # self._init_debug_csv()
        
    def _init_debug_csv(self):
        import pandas as pd
        motor_log_file=[
            f"source/extensions/omni.isaac.lab_tasks/omni/isaac/lab_tasks/direct/quadcopter/motor_{i}_log.csv" for i in range(4)
        ]    
        motor_ref_vel=[]
        DROP_FRONT_LINE=0
        import pandas as pd
        for log_fil in motor_log_file:
            df=pd.read_csv(log_fil)
            # drop front line
            valid_data=df["motor_ref_vel_0"][DROP_FRONT_LINE:].values
            motor_ref_vel_tensor=torch.tensor(valid_data, dtype=torch.float32, device=self.device)
            motor_ref_vel.append(motor_ref_vel_tensor)

        # cat all the tensor
        self.motor_ref_vel=torch.stack(motor_ref_vel, dim=1).to(self.device)
        self.ref_step_cnt=0

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        self.cfg.terrain.num_envs = self.scene.cfg.num_envs
        self.cfg.terrain.env_spacing = self.scene.cfg.env_spacing
        self._terrain = self.cfg.terrain.class_type(self.cfg.terrain)
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)

    def _pre_physics_step(self, actions: torch.Tensor):     
        self.target_rate = actions[:, 0:3].clip(-1,1) * torch.pi
        self.target_thrust = actions[:, 3].clip(0,1).unsqueeze(1)
        if self.cfg.domain_randomization.action.enable:
            self.target_thrust = self.add_multiplicative_noise(self.target_thrust, self.cfg.domain_randomization.action.scale.thrust_scalar)  # ±10%
            self.target_rate = self.add_multiplicative_noise(self.target_rate, self.cfg.domain_randomization.action.scale.bodyrate_scalar)      # ±8%
        self.last_action = actions
        # self.rotor_commands = actions.clone().clamp(0,1)

    def _apply_action(self):
        # run controller loop, calculate command for the rotors
        self.rotor_commands = self.rate_controller.run(self.target_rate, self.target_thrust, self._robot.data.root_ang_vel_b, self.dt)
        # step rotor, get rotor thrust and monent
        self.motor_model.calculate_rotor_dynamic(self.rotor_commands)
        # calculate rotor drag, rolling moment
        
        # apply thrust and moment to the robot
        self.body_moment_sum = self.motor_model.rotor_moment.sum(dim=1, keepdim=True) 
        self._robot.set_external_force_and_torque(self.motor_model.rotor_thrust, self.motor_model.rotor_zero_moment, body_ids=self._rotors_id)
        self._robot.set_external_force_and_torque(torch.zeros_like(self.body_moment_sum), -self.body_moment_sum , body_ids=self._body_id)
        self._robot.data.joint_vel_target[:,self._rotor_joint_ids]=self.motor_model.rotor_velocity*self.motor_model.rotor_directions/self.cfg.rotor_speed_discount

    def add_multiplicative_noise(self, value: torch.Tensor, percent: float) -> torch.Tensor:
        # 在 [1-percent, 1+percent] 范围内均匀分布乘法噪声
        scale = 1.0 + (torch.rand_like(value, device=self.device) * 2 - 1) * percent
        return value * scale
    
    def add_gauss_noise(self, value, scale):
        # add a noise to the value from -scale to scale
        return value + torch.randn_like(value, device=self.device) * scale

    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )

        g_proj=self._robot.data.projected_gravity_b
        g_proj=g_proj/torch.linalg.norm(g_proj, dim=1, keepdim=True)
        self._previous_actions = self._actions.clone()
        
        # 添加偏航角信息
        # 计算当前机体x轴方向（世界坐标系）
        # x_axis_b = torch.tensor([1, 0, 0], device=self.device, dtype=torch.float32)
        # x_axis_b = x_axis_b.expand(self.num_envs, 3)
        # heading_vec_w = quat_rotate(self._robot.data.root_quat_w, x_axis_b)
        
        # # 只取xy平面的朝向信息
        # heading_xy = heading_vec_w[:, :2]
        # heading_xy = heading_xy / (torch.linalg.norm(heading_xy, dim=1, keepdim=True) + 1e-6)
            
        
        # 计算当前无人机偏航角
        quat = self._robot.data.root_quat_w
        yaw = torch.atan2(
            2.0 * (quat[:, 0] * quat[:, 3] + quat[:, 1] * quat[:, 2]),
            1.0 - 2.0 * (quat[:, 2] ** 2 + quat[:, 3] ** 2)
        )
        desired_yaw = self._desired_yaw
        yaw_error = yaw - desired_yaw
        yaw_error = (yaw_error + torch.pi) % (2 * torch.pi) - torch.pi  # wrap到[-pi, pi]
        yaw_error_norm = (yaw_error / torch.pi).unsqueeze(1)  # 映射到[-1, 1]
        
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,     # 线速度 [3]
                self._robot.data.root_ang_vel_b,     # 角速度 [3]
                g_proj,                              # 重力方向 [3]
                desired_pos_b,                       # 目标位置 [3]
                # yaw_error_norm,                      # 偏航角误差 [1]
                self.last_action,                    # 上一步动作 [4]
            ],
            dim=-1,
        )
        # 总维度: 3+3+3+3+1+4 = 17
        if self.cfg.domain_randomization.observation.enable:
            obs_noise = torch.cat(
                [
                    torch.randn_like(self._robot.data.root_lin_vel_b)*self.cfg.domain_randomization.observation.scale.root_lin_vel_b,
                    torch.randn_like(self._robot.data.root_ang_vel_b)*self.cfg.domain_randomization.observation.scale.root_ang_vel_b,
                    torch.zeros_like(self._robot.data.projected_gravity_b),
                    torch.zeros_like(desired_pos_b),
                    # torch.zeros_like(yaw_error_norm),
                    torch.zeros_like(self.last_action),
                ],
                dim=-1,
            )
            obs += obs_noise
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        # lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        # ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 2)
        
        # 计算动作差异奖励
        action_diff = torch.sum(torch.square(self.last_action - self._previous_actions), dim=1)
        
        # 悬停奖励 在距离目标位置1米内的速度奖励
        velocity = torch.linalg.norm(self._robot.data.root_vel_w, dim=1)
        hover_mask = (distance_to_goal < 0.2)
        reward_hover = hover_mask * (1 - torch.tanh(velocity / 0.2))
        
        # 目标速度奖励
        # 目标速度奖励（速度接近0.5奖励最大，始终为正）
        desired_speed = 0.5  # 期望速度
        speed_error = torch.abs(velocity - desired_speed)
        reward_speed = torch.exp(-speed_error * 4)  # 误差越小奖励越大，始终为正
        
        rewards = {
            "speed_reward": reward_speed * self.cfg.speed_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_diff": action_diff * self.cfg.action_diff_reward_scale * self.step_dt,
            "reward_hover": reward_hover * self.cfg.hover_reward_scale * self.step_dt,
        }
        
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        
        # Logging
        for key, value in rewards.items():
            if key not in self._episode_sums:
                self._episode_sums[key] = torch.zeros_like(value)
            self._episode_sums[key] += value
        
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        height_died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.15, self._robot.data.root_pos_w[:, 2] > 3.0)
        
        # distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        # distance_died = distance_to_goal > 4.0  # 例如距离目标超过2米就提前终止

        # velocity_magnitude = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1)
        # velocity_died = velocity_magnitude > 20.0
        
        # 创建具体的终止条件tensor而不是None
        lidar_died = torch.zeros_like(height_died)  # 暂时设为全False
        velocity_died = torch.zeros_like(height_died)  # 暂时设为全False
        died = height_died | lidar_died | velocity_died

        return died, height_died, lidar_died, velocity_died, time_out

    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
        final_distance_to_goal = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        ).mean()
        extras = dict()
        for key in self._episode_sums.keys():
            episodic_sum_avg = torch.mean(self._episode_sums[key][env_ids])
            extras["Episode_Reward/" + key] = episodic_sum_avg / self.max_episode_length_s
            self._episode_sums[key][env_ids] = 0.0
        self.extras["log"] = dict()
        self.extras["log"].update(extras)
        extras = dict()
        extras["Episode_Termination/died"] = torch.count_nonzero(self.reset_terminated[env_ids]).item()
        extras["Episode_Termination/height_died"] = torch.count_nonzero(self.reset_height[env_ids]).item()
        extras["Episode_Termination/velocity_died"] = torch.count_nonzero(self.reset_velocity[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        self._robot.reset(env_ids)
        self.motor_model.reset(env_ids)
        self.rate_controller.reset(env_ids)

        self.last_action[env_ids,:] = 0.0
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        
        # ====== Sample new desired position and yaw ======
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-1.0, 1.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.25, 2.0)
        desired_yaw = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(-torch.pi, torch.pi)
        self._desired_yaw[env_ids] = desired_yaw
        
        # ====== Reset robot state ======
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        
        # same_pos_prob = 0.6  # 例如60%概率目标点和初始点一致
        # if random.random() < same_pos_prob:
        #     default_root_state[:, 2] = self._desired_pos_w[env_ids, 2]
        #     # 保持x和y位置与目标位置一致
        #     default_root_state[:, :2] = self._desired_pos_w[env_ids, :2]
        #     # 保持偏航角与目标偏航角一致
        #     yaw = self._desired_yaw[env_ids]
        #     self._yaw[env_ids] = yaw
        #     # Convert yaw to quaternion (assuming roll=0, pitch=0)
        #     root_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        #     default_root_state[:, 3:7] = root_quat      
        # else:
        
        # Randomly initialize x and y positions within the terrain bounds
        default_root_state[:, :2] = torch.zeros_like(default_root_state[:, :2]).uniform_(-1.0, 1.0)
        # Randomly initialize z position between 0.25 and 2.0
        default_root_state[:, 2] = torch.zeros_like(default_root_state[:, 2]).uniform_(0.25, 2.0)
        default_root_state[:, :2] += self._terrain.env_origins[env_ids, :2]
    
        # Set random yaw orientation for each environment
        yaw = torch.zeros_like(default_root_state[:, 2]).uniform_(-torch.pi, torch.pi)
        self._yaw[env_ids] = yaw
        # Convert yaw to quaternion (assuming roll=0, pitch=0)
        root_quat = quat_from_euler_xyz(torch.zeros_like(yaw), torch.zeros_like(yaw), yaw)
        default_root_state[:, 3:7] = root_quat
    
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first time
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
            
            # 添加机头朝向红色箭头可视化
            if not hasattr(self, "heading_visualizer"):
                from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (1.0, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/heading"
                self.heading_visualizer = VisualizationMarkers(marker_cfg)
            self.heading_visualizer.set_visibility(True)
            
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "heading_visualizer"):
                self.heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
        
        # === 显示无人机朝向（机体x轴）蓝色箭头 ===
        base_quat_w = self._robot.data.root_quat_w
        
        # 机体x轴在世界坐标系下的方向
        x_axis_b = torch.tensor([1, 0, 0], device=self.device, dtype=base_quat_w.dtype).expand(self.num_envs, 3)
        heading_vec_w = quat_rotate(base_quat_w, x_axis_b)
        heading_vec_w[:, 2] = 0.0  # 只考虑xy平面的朝向
        
        # 计算箭头的尺寸和方向
        heading_scale = torch.tensor([2.0, 0.15, 0.15], device=self.device).repeat(heading_vec_w.shape[0], 1)
        heading_scale[:, 0] *= torch.linalg.norm(heading_vec_w[:, :2], dim=1) * 2.0
        
        # 计算朝向角度并生成四元数
        heading_angle = torch.atan2(heading_vec_w[:, 1], heading_vec_w[:, 0])
        zeros = torch.zeros_like(heading_angle)
        heading_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        # 箭头位置：无人机位置稍微向上偏移
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.1
        
        # 可视化蓝色朝向箭头
        self.heading_visualizer.visualize(base_pos_w, heading_quat, heading_scale)