# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg, RigidObject, RigidObjectCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

from isaaclab_tasks.direct.quadcopter.modules.controller import RateController
from isaaclab_tasks.direct.quadcopter.modules.motor import MotorModel
from isaaclab.sensors import RayCaster
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate
from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG

##
# Pre-defined configs
##
from isaaclab_assets import UAVLIDAR_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .lidarfly_cfg import LidarFlyEnvCfg, LidarDeploymentEnvCfg
from collections import deque

class LidarDeploymentEnv(DirectRLEnv):
    cfg: LidarDeploymentEnvCfg

    def __init__(self, cfg: LidarDeploymentEnvCfg, render_mode: str | None = None, **kwargs):
        super().__init__(cfg, render_mode, **kwargs)
        self._died_height = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_lidar = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._died_velocity = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        
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
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "ang_vel",
                "z",
                "action_diff",
                "live",
                "reward_dir",
                "reward_yaw",
                "reward_distance",
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
        
        self.current_scan = torch.zeros(self.num_envs, self.cfg.lidar_resolution[0]*self.cfg.lidar_resolution[1], device=self.device).reshape(self.num_envs, -1)
        self.episode_count = 0
        self.success_count = 0
        self.success_window = deque(maxlen=100)  # 统计最近100个episode的成功情况
        
        # yaw init
        self.heading_vec_w = torch.zeros(self.num_envs, 3, device=self.device)  # 用于存储无人机机体x轴在世界坐标系下的方向
        self.heading_xy = torch.zeros(self.num_envs, 2, device=self.device)  # 用于存储无人机机体x轴在xy平面上的方向
        self.target_direction_xy = torch.zeros(self.num_envs, 2, device=self.device)  # 用于存储目标方向在xy平面上的方向
        # obs 3 -
        # 初始化队列来存储最近的3帧观测数据
        self.obs_queue = deque(maxlen=3)
        initial_obs_shape = (self.num_envs, self._robot.data.root_lin_vel_b.shape[-1] + 
                             self._robot.data.root_ang_vel_b.shape[-1] + 1 +
                             1 + 3 + 2 + 1 +
                             self.last_action.shape[-1])
        initial_obs = torch.zeros(initial_obs_shape, device=self.device)
        for _ in range(3):
            self.obs_queue.append(initial_obs)
        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        # self._init_debug_csv()
        
    def _extract_euler_angles(self, quat):
        """从四元数提取欧拉角 (roll, pitch, yaw)"""
        # quat: [w, x, y, z]
        w, x, y, z = quat[:, 0], quat[:, 1], quat[:, 2], quat[:, 3]
        
        # Roll (x轴旋转)
        sinr_cosp = 2 * (w * x + y * z)
        cosr_cosp = 1 - 2 * (x * x + y * y)
        roll = torch.atan2(sinr_cosp, cosr_cosp)
        
        # Pitch (y轴旋转)
        sinp = 2 * (w * y - z * x)
        pitch = torch.where(
            torch.abs(sinp) >= 1,
            torch.sign(sinp) * (torch.pi / 2),  # 处理gimbal lock
            torch.asin(sinp)
        )
        
        # Yaw (z轴旋转) - 你已经在用其他方法计算yaw，这里可以不用
        siny_cosp = 2 * (w * z + x * y)
        cosy_cosp = 1 - 2 * (y * y + z * z)
        yaw = torch.atan2(siny_cosp, cosy_cosp)
        
        return roll, pitch, yaw

    def _setup_scene(self):
        self._robot = Articulation(self.cfg.robot)
        self.scene.articulations["robot"] = self._robot

        # ray caster  
        self.ray_caster = RayCaster(self.cfg.ray_scanner)
        self.scene.sensors["ray_scanner"] = self.ray_caster
        
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
        # return
        self.target_rate = actions[:, 0:3].clip(-1,1) * torch.pi
        self.target_thrust = actions[:, 3].clip(0,1).unsqueeze(1)
        self.last_action = actions.clip(-1,1)

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
    
    def add_guass_noise(self,value,base_scale):
        # 根据距离值动态调整噪声尺度
        dynamic_scale = base_scale * (1 - value)
        # 生成与 value 形状相同的高斯噪声，并将其限制在 [-0.5, 0.5] 范围内
        noise = torch.randn_like(value, device=self.device).clip(-0.5, 0.5) * dynamic_scale
        return value + noise
        
    def _get_observations(self) -> dict:
        self._previous_actions = self._actions.clone()
        # ===== 基础计算 =====
        base_quat_w = self._robot.data.root_quat_w
        
        # ===== 提取roll和pitch角度 =====
        roll, pitch, yaw = self._extract_euler_angles(base_quat_w)
        # 归一化到 [-1, 1] 范围
        roll_normalized = roll / torch.pi
        pitch_normalized = pitch / torch.pi
        
        # ===== Z轴误差 =====
        z_error = (self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2]).unsqueeze(-1)

        # ===== XY方向的单位向量 =====
        target_direction_w = self._desired_pos_w - self._robot.data.root_pos_w
        target_direction_xy = target_direction_w[:, :2]
        target_distance_xy = torch.linalg.norm(target_direction_xy, dim=1, keepdim=True)
        target_direction_xy_normalized = target_direction_xy / (target_distance_xy + 1e-6)
        
        # ===== 偏航角误差计算（直接使用已计算的yaw） =====
        target_yaw = torch.atan2(target_direction_w[:, 1], target_direction_w[:, 0])
        yaw_error = target_yaw - yaw  # 直接使用前面计算的yaw
        # 处理角度环绕问题
        yaw_error = torch.atan2(torch.sin(yaw_error), torch.cos(yaw_error))
        yaw_error_normalized = yaw_error / torch.pi
        
        # 存储供奖励函数使用
        self.current_yaw = yaw
        self.target_yaw = target_yaw
        self.yaw_error = yaw_error
        
        # ===== 激光雷达处理（保持原有逻辑）=====
        current_lidar_hits = self.scene["ray_scanner"].data.ray_hits_w
        current_lidar_pos = self.ray_caster.data.pos_w.unsqueeze(1)
        
        self.current_scan = self.cfg.lidar_range - (
            (current_lidar_hits - current_lidar_pos)
            .norm(dim=-1)
            .clamp_max(self.cfg.lidar_range)
            .reshape(self.num_envs, 1, *self.cfg.lidar_resolution)
        ).reshape(self.num_envs, -1)

        self.current_scan = self.current_scan / self.cfg.lidar_range
        self.current_scan_noise = self.current_scan
        
        if self.cfg.domain_randomization.lidar_noise.enable:
            self.current_scan_noise = self.add_guass_noise(self.current_scan, self.cfg.domain_randomization.lidar_noise.noise)
        
        # ===== 修改观测组合（使用roll、pitch、yaw_error）=====
        current_non_lidar_obs = torch.cat([
            self._robot.data.root_lin_vel_b / 5,               # 3维：线速度
            self._robot.data.root_ang_vel_b,                   # 3维：角速度
            z_error / 2,                                       # 1维：Z高度误差
            roll_normalized.unsqueeze(-1),                     # 1维：roll角度
            pitch_normalized.unsqueeze(-1),                    # 1维：pitch角度
            yaw_error_normalized.unsqueeze(-1),                # 1维：偏航角误差（不用单独的yaw）
            target_direction_xy_normalized,                    # 2维：XY方向单位向量
            target_distance_xy.squeeze(-1).unsqueeze(-1) / 10, # 1维：XY距离
            self.last_action,                                  # 4维：上一步动作
        ], dim=-1)  # 总共 17维
        
        # 去掉历史队列，简化观测
        obs = torch.cat([
            self.current_scan_noise,
            current_non_lidar_obs,
        ], dim=-1)
        obs = obs.clip(-5, 5)
        
        critic_obs = torch.cat([
            self.current_scan,
            current_non_lidar_obs,
        ], dim=-1)
        critic_obs = critic_obs.clip(-5, 5)
        
        observations = {"policy": obs, "critic": critic_obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        action_diff = torch.sum(torch.square(self.last_action - self._previous_actions), dim=1)
        # vel reward
        vel_direction = (self._desired_pos_w - self._robot.data.root_pos_w)
        vel_direction = vel_direction / torch.norm(vel_direction, dim=-1, keepdim=True)
        reward_dir = (self._robot.data.root_lin_vel_w * vel_direction).sum(-1).clip(max=3.0)
        reward_z = torch.exp(-5 * torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2]))
         
        # ------------------------------- forward facing reward -------------------------------
        # ===== 更简洁的偏航奖励 =====
        # 直接使用偏航角误差，越小越好
        reward_yaw = torch.exp(-2 * torch.abs(self.yaw_error))  # 偏航误差越小，奖励越高
        # ---------------------------------------------------------------
        
        distance_to_goal = torch.linalg.norm(
            self._desired_pos_w - self._robot.data.root_pos_w, dim=1
        )
        reward_distance = torch.exp(-2 * distance_to_goal)  # 距离目标越近，奖励越高  
        
        # lidar 
        live = torch.ones_like(lin_vel)
        rewards = {
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "z": reward_z * self.cfg.z_reward_scale * self.step_dt,
            "action_diff" : action_diff* self.cfg.action_diff_reward_scale * self.step_dt,
            "live" : self.cfg.live_scale * live * self.step_dt,
            "reward_dir": reward_dir * self.cfg.dir_reward_scale * self.step_dt,
            "reward_yaw": reward_yaw * self.cfg.yaw_reward_scale * self.step_dt,
            "reward_distance": reward_distance * self.cfg.distance_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward
    
    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        # height_died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.25, self._robot.data.root_pos_w[:, 2] > 3.5)
        height_died = torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2]) > 0.5
        lidar_died = torch.any(self.current_scan > (self.cfg.lidar_range - 0.35)/self.cfg.lidar_range, dim=1)
        velocity_magnitude = torch.linalg.norm(self._robot.data.root_lin_vel_w, dim=1)
        # acc_magnitude = torch.linalg.norm(self._robot.data.body_lin_acc_w, dim=1)
        
        velocity_died = velocity_magnitude > 3.0
        
        died = height_died | lidar_died | velocity_died
        # print("current_scan", self.current_scan)
        # if height_died.any():
        #     print("height_died")
        # if lidar_died.any():
        #     print("lidar_died")
        # if velocity_died.any():
        #     print("velocity_died")
        # if time_out.any():
        #     print("time_out")
        return died, height_died, lidar_died, velocity_died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES

        # Logging
            
        # 计算每个环境的距离
        final_distance = torch.linalg.norm(
            self._desired_pos_w[env_ids] - self._robot.data.root_pos_w[env_ids], dim=1
        )

        # 判断哪些是成功（未died且未timeout，且距离目标小于阈值）
        # 只统计本次真正终止的环境
        # 只统计本次reset的环境中，未died、未timeout且到达目标的为成功
        # success_mask = (~self.reset_terminated[env_ids]) & (~self.reset_time_outs[env_ids]) & (final_distance < 1.2)
        success_mask = (final_distance < 2)
        self.success_window.extend(success_mask.cpu().numpy().tolist())
        success_rate = sum(self.success_window) / max(1, len(self.success_window))
        
        # ...原有日志统计...
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
        extras["Episode_Termination/lidar_died"] = torch.count_nonzero(self.reset_lidar[env_ids]).item()
        extras["Episode_Termination/time_out"] = torch.count_nonzero(self.reset_time_outs[env_ids]).item()
        extras["Metrics/final_distance_to_goal"] = final_distance_to_goal.item()
        self.extras["log"].update(extras)

        # 写入滑动窗口成功率
        self.extras["log"]["Episode_Success/success_rate"] = success_rate
            
        self._robot.reset(env_ids)
        self.motor_model.reset(env_ids)
        self.rate_controller.reset(env_ids)

        self.last_action[env_ids,:] = 0.0
        super()._reset_idx(env_ids)
        if len(env_ids) == self.num_envs:
            # Spread out the resets to avoid spikes in training when many environments reset at a similar time
            self.episode_length_buf = torch.randint_like(self.episode_length_buf, high=int(self.max_episode_length))

        self._actions[env_ids] = 0.0
        # Sample new commands
        self._desired_pos_w[env_ids, 0] = torch.zeros_like(self._desired_pos_w[env_ids, 0]).uniform_(7, 8.5)
        self._desired_pos_w[env_ids, 1] = torch.zeros_like(self._desired_pos_w[env_ids, 1]).uniform_(-3, 3)
        self._desired_pos_w[env_ids, :2] += self._terrain.terrain_origins.view(-1,3)[env_ids, :2]
        # print("terrain origins", self._terrain.terrain_origins)
        # print("terrain env origins", self._terrain.env_origins)
        # print("self._desired_pos_w", self._desired_pos_w)
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.8, 1.2)
        # y_coords = torch.linspace(-self.cfg.y_field, self.cfg.y_field, steps=self.num_envs, device=self.device) 
        # self._desired_pos_w[env_ids, 0] = self._robot.data.default_root_state[env_ids][:, 0] + 20
        # self._desired_pos_w[env_ids, 1] = y_coords[env_ids]
        # self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 2)
        
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        # default_root_state = self._robot.data.default_root_state[env_ids]
        # # 根据环境标号分配 y 坐标
        # default_root_state[:, 1] = y_coords[env_ids]
        
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.terrain_origins.view(-1,3)[env_ids]
        

        # print(f"初始高度: {default_root_state[:, 2]}, 期望高度: {self._desired_pos_w[:, 2]}, 差值: {torch.abs(default_root_state[:, 2] - self._desired_pos_w[:, 2])}")
        # Randomize the orientation of the drone within the front 180 degrees
        # random_yaw = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        # random_quaternions = torch.stack([
        #     torch.cos(random_yaw / 2),
        #     torch.zeros_like(random_yaw),
        #     torch.zeros_like(random_yaw),
        #     torch.sin(random_yaw / 2)
        # ], dim=-1)
        # default_root_state[:, 3:7] = random_quaternions
        default_root_state[:, 3:7] = torch.tensor([1.0, 0.0, 0.0, 0.0], device=self.device).expand(len(env_ids), 4)
        
        # # 确保 default_root_state 的形状正确
        # assert default_root_state.shape[1] >= 7, "default_root_state 的形状不正确"

        # # 打印 env_ids 的值
        # print(f"env_ids: {env_ids.cpu()}")
        # # 确保 env_ids 的范围有效
        # assert env_ids.max() < 1024, "env_ids 超出范围"
        
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

        # === 显示无人机当前yaw朝向（机体x轴）蓝色箭头 ===
        # 使用 self.current_yaw 计算朝向
        heading_angle = self.current_yaw  # [num_envs]
        zeros = torch.zeros_like(heading_angle)
        heading_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)

        # 箭头位置：无人机位置稍微向上偏移
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.1

        # 箭头尺寸
        heading_scale = torch.tensor([2.0, 0.15, 0.15], device=self.device).repeat(base_pos_w.shape[0], 1)

        # 可视化蓝色朝向箭头
        self.heading_visualizer.visualize(base_pos_w, heading_quat, heading_scale)
