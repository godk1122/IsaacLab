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

from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG, BLUE_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate
##
# Pre-defined configs   
##
from isaaclab_assets import UAVLIDAR_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .lidarguide_cfg import LidarGuideEnvCfg
from collections import deque

def quat_inverse(q):
    # q: [num_envs, 4]
    q_inv = q.clone()
    q_inv[:, :3] *= -1
    return q_inv

def detect_wall_continuous(scan_slice, threshold=0.3, window_size=5, min_count=5):
    # scan_slice: [num_envs, 36]，已对竖直分辨率做均值
    hits = (scan_slice > threshold).float()  # [num_envs, 36]
    # 构造滑动窗口和卷积核
    kernel = torch.ones(window_size, device=scan_slice.device)
    # 用1D卷积检测连续障碍数量
    conv = torch.nn.functional.conv1d(
        hits.unsqueeze(1),  # [num_envs, 1, 36]
        kernel.view(1, 1, -1),  # [1, 1, window_size]
        padding=window_size // 2
    ).squeeze(1)  # [num_envs, 36]
    # 返回每个方向窗口是否满足条件
    wall_mask = (conv >= min_count)  # [num_envs, 36]，每个方向 True/False
    return wall_mask

def detect_wall(current_scan, desired_p_b):
    # current_scan: [num_envs, 72, 2]
    num_envs, num_rays, num_z = current_scan.shape
    half_rays = num_rays // 2  # 180度对应36个

    # 计算目标方向角
    target_angle = torch.atan2(desired_p_b[:, 1], desired_p_b[:, 0])  # [num_envs]
    num_rays = current_scan.shape[1]
    # 先生成 [0, π]，再生成 [-π, 0)
    ray_angles_pos = torch.linspace(0, torch.pi, num_rays // 2, device=current_scan.device, dtype=current_scan.dtype)
    ray_angles_neg = torch.linspace(-torch.pi, 0, num_rays // 2, device=current_scan.device, dtype=current_scan.dtype)[:-1]  # 去掉0避免重复
    ray_angles = torch.cat([ray_angles_pos, ray_angles_neg], dim=0)  # [num_rays]
    angle_diff = torch.abs(ray_angles.unsqueeze(0) - target_angle.unsqueeze(1))  # [num_envs, 72]
   
    center_idx = angle_diff.argmin(dim=1)  # [num_envs]
    # 构造每个环境的180度索引（环形索引）
    idx = (torch.arange(-half_rays//2, half_rays//2, device=current_scan.device).unsqueeze(0) + center_idx.unsqueeze(1)) % num_rays  # [num_envs, 36]
    scan_slice = current_scan[torch.arange(num_envs).unsqueeze(1), idx, :]  # [num_envs, 36, 2]
    scan_slice_mean = scan_slice.mean(dim=2)  # [num_envs, 36]
    wall_mask = detect_wall_continuous(scan_slice_mean, threshold=0.3, window_size=5, min_count=3)
    return wall_mask, target_angle


class LidarGuideRnnEnv(DirectRLEnv):
    cfg: LidarGuideEnvCfg

    def __init__(self, cfg: LidarGuideEnvCfg, render_mode: str | None = None, **kwargs):
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
        self.optimal_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.dt = self.cfg.sim.dt
        self.motor_model = MotorModel(self.num_envs, self.device, self.dt, self.cfg.domain_randomization.motor)
        self.rate_controller = RateController(self.num_envs, self.device)
        
        self.last_action = torch.zeros(self.num_envs, 4, device=self.device)
        # self.step_dt=self.dt*self.cfg.decimation
        # Logging
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "z",
                "action_diff",
                "live",
                "reward_dir",
                "reward_g_proj",
                "reward_yaw",
                "direction_change",
                "reward_distance"
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
        # 在 __init__ 里初始化
        self.prev_optimal_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.optimal_direction_alpha = 0.7  # 平滑系数，可调
        self.optimal_dir_update_interval = 5  # 每5步更新一次，可调

        # add handle for debug visualization (this is set to a valid handle inside set_debug_vis)
        self.set_debug_vis(self.cfg.debug_vis)
        # self._init_debug_csv()
    def _init_debug_csv(self):
        import pandas as pd
        motor_log_file=[
            f"source/extensions/isaaclab_tasks/omni/isaac/lab_tasks/direct/quadcopter/motor_{i}_log.csv" for i in range(4)
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
        #  set controller target here
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
    
   
    def get_optimal_direction(self, desired_p_b):
        # 获取当前帧的激光雷达数据
        current_lidar_hits = self.scene["ray_scanner"].data.ray_hits_w
        current_lidar_pos = self.ray_caster.data.pos_w.unsqueeze(1)
        current_scan = self.cfg.lidar_range - (
            (current_lidar_hits - current_lidar_pos)
            .norm(dim=-1)
            .clamp_max(self.cfg.lidar_range)
            .reshape(self.num_envs, 1, *self.cfg.lidar_resolution)
        )
        
        current_scan = current_scan / self.cfg.lidar_range  # [num_envs, 1, 72, 5]
        current_scan = current_scan.squeeze(1)  # [num_envs, 72, 5]
        
        # 只取中间两个竖直分辨率
        scan_detect = current_scan[..., 1:3]

        # 判断前方是否为墙壁
        wall_mask, target_angle = detect_wall(scan_detect, desired_p_b)  # [num_envs, 36(des_pos_b方向的180度)]
        need_avoid = wall_mask.any(dim=-1)  # [num_envs]，如果有需要避让的方向

        # 默认直接朝向目标
        optimal_direction = desired_p_b.clone()

        if need_avoid.any():
            # wall_mask: [num_envs, 36]，每个方向 True/False
            num_envs, num_dirs = wall_mask.shape
            center_idx = num_dirs // 2  # 中心索引（即目标方向）

            # 对每个环境，找到最近的 False 索引
            false_mask = ~wall_mask  # [num_envs, 36]
            indices = torch.arange(num_dirs, device=wall_mask.device).unsqueeze(0).expand(num_envs, -1)  # [num_envs, 36]
            center_idx = num_dirs // 2  # 中心索引

            # 左侧（小于center_idx）最近的False
            left_mask = (indices < center_idx) & false_mask  # [num_envs, 36]
            left_dist = torch.where(left_mask, center_idx - indices, torch.full_like(indices, 1e6))  # [num_envs, 36]
            left_nearest_idx = left_dist.argmin(dim=1)  # [num_envs]
            # 右侧（大于center_idx）最近的False
            right_mask = (indices > center_idx) & false_mask  # [num_envs, 36]
            right_dist = torch.where(right_mask, indices - center_idx, torch.full_like(indices, 1e6))  # [num_envs, 36]
            right_nearest_idx = right_dist.argmin(dim=1)  # [num_envs]
            
            # 计算最近可行方向与中心的偏移量（以索引为单位）
            left_offset_indices = center_idx - left_nearest_idx  # [num_envs]
            right_offset_indices = center_idx - right_nearest_idx  # [num_envs]
            
            # 每个索引对应的角度偏移量
            angle_per_index = 2 * torch.pi / 72
            new_left_target_angle = target_angle + left_offset_indices * angle_per_index  # [num_envs]
            new_right_target_angle = target_angle + right_offset_indices * angle_per_index  # [num_envs]
            
            # ---------------- 左右选择 ----------------
            # 反向计算 optimal_direction 的 xy 分量
            chosen_left_dir = torch.stack([torch.cos(new_left_target_angle), torch.sin(new_left_target_angle)], dim=-1)  # [num_envs, 2]
            chosen_right_dir = torch.stack([torch.cos(new_right_target_angle), torch.sin(new_right_target_angle)], dim=-1)  # [num_envs, 2]
    
            # 将 chosen_left_dir 和 chosen_right_dir 从世界坐标系转换为机体坐标系
            base_quat_w = self._robot.data.root_quat_w
            chosen_left_dir_b = quat_rotate(quat_inverse(base_quat_w), torch.cat([chosen_left_dir, torch.zeros(num_envs, 1, device=self.device)], dim=-1))[:, :2]
            chosen_right_dir_b = quat_rotate(quat_inverse(base_quat_w), torch.cat([chosen_right_dir, torch.zeros(num_envs, 1, device=self.device)], dim=-1))[:, :2]            # 计算无人机机头（x轴）在机体坐标系下的方向
            
            x_axis_b = torch.tensor([1, 0], device=self.device, dtype=chosen_left_dir_b.dtype).expand(num_envs, 2)
            # 计算 chosen_left_dir_b 和 chosen_right_dir_b 与机头方向的夹角
            left_dot = (chosen_left_dir_b * x_axis_b).sum(dim=-1)
            right_dot = (chosen_right_dir_b * x_axis_b).sum(dim=-1)
            # 选择与机头方向夹角更小（dot更大）的方向
            cond = (left_dot > right_dot).unsqueeze(1)  # [num_envs, 1]
            chosen_dir = torch.where(cond, chosen_left_dir_b, chosen_right_dir_b)  # [num_envs, 2]
            # ---------------------------------------
            
            # 对 optimal_direction 做 xy 平面偏移
            optimal_direction = desired_p_b.clone()
            optimal_direction[:, :2] = chosen_dir
            optimal_direction = optimal_direction / (optimal_direction.norm(dim=-1, keepdim=True) + 1e-6)

        # 归一化
        optimal_direction = optimal_direction / (optimal_direction.norm(dim=-1, keepdim=True) + 1e-6)
        return optimal_direction
        
    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        # 计算单位方向向量
        desired_pos_b  = desired_pos_b / torch.norm(desired_pos_b, dim=-1, keepdim=True)
        
        g_proj=self._robot.data.projected_gravity_b
        g_proj=g_proj/torch.linalg.norm(g_proj, dim=1, keepdim=True)
        self._previous_actions = self._actions.clone()
        # lidar 
        # 获取当前帧的激光雷达数据
        current_lidar_hits = self.scene["ray_scanner"].data.ray_hits_w
        current_lidar_pos = self.ray_caster.data.pos_w.unsqueeze(1)
        
        self.current_scan = self.cfg.lidar_range - (
            (current_lidar_hits - current_lidar_pos)
            .norm(dim=-1)
            .clamp_max(self.cfg.lidar_range)
            .reshape(self.num_envs, 1, *self.cfg.lidar_resolution)
        ).reshape(self.num_envs, -1)

        self.current_scan = self.current_scan / self.cfg.lidar_range
        self.current_scan_noise  = self.current_scan
        
        # lidar domain randomization
        if self.cfg.domain_randomization.lidar_noise.enable:
            self.current_scan_noise = self.add_guass_noise(self.current_scan, self.cfg.domain_randomization.lidar_noise.noise)
        
        # 确保 self._robot.data.root_state_w[:, 2] 的维度与其他张量相同
        root_state_w_z = (self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2]).unsqueeze(-1)
        current_non_lidar_obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b / 5,
                self._robot.data.root_ang_vel_b,
                root_state_w_z/2,
                g_proj,
                desired_pos_b,
                self.last_action,
            ],
            dim=-1,
        )
        
        # 将激光雷达部分与累积的非激光雷达部分观测数据拼接
        obs = torch.cat(
            [
                self.current_scan_noise,
                current_non_lidar_obs,
            ],
            dim=-1,
        )
        obs= obs.clip(-2,2)
        
        critic_obs = torch.cat(
            [
                self.current_scan,
                self._robot.data.root_lin_vel_b / 5,
                self._robot.data.root_ang_vel_b,
                root_state_w_z/2,
                self._robot.data.root_pos_w[:, 2].unsqueeze(-1),
                g_proj,
                desired_pos_b,
                self.optimal_direction,
                self.last_action,
            ],
            dim=-1,
        )
        critic_obs= critic_obs.clip(-2,2)
        
        if self.cfg.domain_randomization.noise.enable:
            obs_noise = torch.cat(
                [
                    torch.randn_like(self._robot.data.root_lin_vel_b)*self.cfg.domain_randomization.noise.root_lin_vel_b,
                    torch.randn_like(self._robot.data.root_ang_vel_b)*self.cfg.domain_randomization.noise.root_ang_vel_b,
                    torch.zeros_like(self._robot.data.projected_gravity_b),
                    torch.zeros_like(desired_pos_b),
                    torch.zeros_like(self.last_action),
                ],
                dim=-1,
            )
            obs += obs_noise
            
        observations = {"policy": obs,
                        "critic": critic_obs}
        return observations
    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        action_diff = torch.sum(torch.square(self.last_action - self._previous_actions), dim=1)
        
        # ----------------------------------------------------------------
        # get optimal direction
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        # 计算单位方向向量
        desired_pos_b  = desired_pos_b / torch.norm(desired_pos_b, dim=-1, keepdim=True)
        # 控制optimal_direction的更新频率
        if not hasattr(self, "optimal_dir_update_counter"):
            self.optimal_dir_update_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        if not hasattr(self, "optimal_dir_update_interval"):
            self.optimal_dir_update_interval = 5  # 可调
        update_mask = (self.optimal_dir_update_counter % self.optimal_dir_update_interval == 0)
        if not hasattr(self, "optimal_direction"):
            self.optimal_direction = desired_pos_b.clone()
        if update_mask.any():
            new_optimal_direction = self.get_optimal_direction(desired_pos_b)
            self.optimal_direction[update_mask] = new_optimal_direction[update_mask]
        self.optimal_dir_update_counter += 1

        # 如果距离终点较近（例如小于0.8米），则 optimal_direction 直接等于 desired_pos_b
        close_to_goal = distance_to_goal < 4
        self.optimal_direction = torch.where(
            close_to_goal.unsqueeze(-1),
            desired_pos_b,
            self.optimal_direction
        )
        # ---------------------------------------------------------------
        # vel reward
        vel_direction = (self._desired_pos_w - self._robot.data.root_pos_w)
        vel_direction = vel_direction / torch.norm(vel_direction, dim=-1, keepdim=True)
        reward_dir = (self._robot.data.root_lin_vel_b * self.optimal_direction).sum(-1).clip(max=5.0)
        reward_z = torch.exp(-5 * torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2]))
        
        g_proj = self._robot.data.projected_gravity_b
        g_proj = g_proj / torch.linalg.norm(g_proj, dim=1, keepdim=True)
        # Reward for keeping the drone stable (aligned with gravity)
        g_proj_reward = torch.exp(-5 * torch.abs(-1 - g_proj[:, 2]))
        
        # reward_esdf = torch.exp(-5 * self.current_scan.max(dim=1).values)
        
        # ------------------------------- forward facing reward -------------------------------
        # 奖励无人机始终保持“向前”姿态（即机体x轴始终朝向世界坐标系x轴正方向）
        # 机体x轴在世界坐标系下的方向
        base_quat_w = self._robot.data.root_quat_w
        x_axis_b = torch.tensor([1, 0, 0], device=self.device, dtype=base_quat_w.dtype).expand(self.num_envs, 3)
        heading_vec_w = quat_rotate(base_quat_w, x_axis_b)  # [num_envs, 3]
        v = self._robot.data.root_lin_vel_w  # [num_envs, 3]
        v_norm = v.norm(dim=1, keepdim=True) + 1e-6
        reward_yaw = (heading_vec_w * v).sum(dim=1) / v_norm.squeeze(1)  # [num_envs]
        # ---------------------------------------------------------------
        
        # optimal dir smooth reward 
        if not hasattr(self, "prev_optimal_direction"):
            self.prev_optimal_direction = self.optimal_direction.clone()
        direction_change = torch.norm(self.optimal_direction - self.prev_optimal_direction, dim=1)
        direction_change_penalty = -1 * direction_change  # 系数可调
        self.prev_optimal_direction = self.optimal_direction.clone()
        
        # --------------------- 终点距离奖励 ---------------------
        # 奖励无人机接近目标位置
        reward_distance = 5 * torch.exp(-0.2 * distance_to_goal)  
        
        # lidar 
        live = torch.ones_like(lin_vel)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "z": reward_z * self.cfg.z_reward_scale * self.step_dt,
            "action_diff" : action_diff* self.cfg.action_diff_reward_scale * self.step_dt,
            "live" : self.cfg.live_scale * live * self.step_dt,
            "reward_dir": reward_dir * self.cfg.dir_reward_scale * self.step_dt,
            "reward_g_proj": g_proj_reward * self.cfg.g_proj_reward_scale * self.step_dt,
            "reward_yaw": reward_yaw * self.cfg.reward_forward_facing_scale * self.step_dt,
            "direction_change": direction_change_penalty * self.step_dt,
            "reward_distance": reward_distance * self.cfg.reward_distance_scale * self.step_dt,
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
        
        velocity_died = velocity_magnitude > 5.0
        
        died = height_died | lidar_died | velocity_died

        return died, height_died, lidar_died, velocity_died, time_out


    def _reset_idx(self, env_ids: torch.Tensor | None):
        if env_ids is None or len(env_ids) == self.num_envs:
            env_ids = self._robot._ALL_INDICES
      
        # --- 新增：reset optimal_direction 相关变量 ---
        # 重新计算 optimal_direction
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[env_ids, :3], self._robot.data.root_state_w[env_ids, 3:7], self._desired_pos_w[env_ids]
        )
        desired_pos_b = desired_pos_b / (torch.norm(desired_pos_b, dim=-1, keepdim=True) + 1e-6)
        if not hasattr(self, "optimal_direction"):
            self.optimal_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.optimal_direction[env_ids] = desired_pos_b

        # reset prev_optimal_direction
        if not hasattr(self, "prev_optimal_direction"):
            self.prev_optimal_direction = torch.zeros(self.num_envs, 3, device=self.device)
        self.prev_optimal_direction[env_ids] = desired_pos_b

        # reset optimal_dir_update_counter
        if not hasattr(self, "optimal_dir_update_counter"):
            self.optimal_dir_update_counter = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.optimal_dir_update_counter[env_ids] = 0
        # --------------------------------------------

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
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.2, 0.2, 0.2)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
          
            if not hasattr(self, "direction_visualizer"):
                marker_cfg = RED_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (1.0, 0.1, 0.1)  # 设置箭头的大小
                # -- direction pose
                marker_cfg.prim_path = "/Visuals/Command/direction"
                self.direction_visualizer = VisualizationMarkers(marker_cfg)
             # set their visibility to true
            self.direction_visualizer.set_visibility(True)
            
            if not hasattr(self, "heading_visualizer"):
                marker_cfg = BLUE_ARROW_X_MARKER_CFG.copy()
                marker_cfg.markers["arrow"].scale = (1.0, 0.1, 0.1)
                marker_cfg.prim_path = "/Visuals/Command/heading"
                self.heading_visualizer = VisualizationMarkers(marker_cfg)
            self.heading_visualizer.set_visibility(True)
            
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)
            if hasattr(self, "direction_visualizer"):
                self.direction_visualizer.set_visibility(False)
    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
       
        # 将 optimal_direction 从机体坐标系转换到世界坐标系
        base_quat_w = self._robot.data.root_quat_w
        optimal_direction_w = quat_rotate(base_quat_w, self.optimal_direction)
        
        # 确保转换到世界坐标系后的 z 轴方向为 0
        optimal_direction_w[:, 2] = 0.0
        
        # 计算方向箭头的比例和四元数
        arrow_scale = torch.tensor([3.0, 0.3, 0.3], device=self.device).repeat(optimal_direction_w.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(optimal_direction_w[:, :2], dim=1) * 3.0
        heading_angle = torch.atan2(optimal_direction_w[:, 1], optimal_direction_w[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        
        # 更新方向箭头的位置和方向
        base_pos_w = self._robot.data.root_pos_w.clone()
        base_pos_w[:, 2] += 0.1
        self.direction_visualizer.visualize(base_pos_w, arrow_quat, arrow_scale)
        
        # === 新增：无人机朝向（机体x轴）蓝色箭头 ===
        # 机体x轴在世界坐标系下的方向
        x_axis_b = torch.tensor([1, 0, 0], device=self.device, dtype=base_quat_w.dtype).expand(self.num_envs, 3)
        heading_vec_w = quat_rotate(base_quat_w, x_axis_b)
        heading_vec_w[:, 2] = 0.0
        heading_scale = torch.tensor([2.0, 0.15, 0.15], device=self.device).repeat(heading_vec_w.shape[0], 1)
        heading_scale[:, 0] *= torch.linalg.norm(heading_vec_w[:, :2], dim=1) * 2.0
        heading_angle = torch.atan2(heading_vec_w[:, 1], heading_vec_w[:, 0])
        heading_quat = quat_from_euler_xyz(zeros, zeros, heading_angle)
        self.heading_visualizer.visualize(base_pos_w, heading_quat, heading_scale)