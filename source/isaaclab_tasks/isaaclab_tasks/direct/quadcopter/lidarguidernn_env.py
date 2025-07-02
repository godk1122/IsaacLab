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

from isaaclab.markers.config import RED_ARROW_X_MARKER_CFG
from isaaclab.utils.math import quat_from_euler_xyz, quat_rotate
##
# Pre-defined configs   
##
from isaaclab_assets import UAVLIDAR_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from .lidarguide_cfg import LidarGuideEnvCfg
from collections import deque

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
                # "esdf",
                # "distance_to_goal",
                "action_diff",
                "live",
                "reward_dir",
                "reward_g_proj",
                "reward_yaw",
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
        # obs 3 -
        # 初始化队列来存储最近的3帧观测数据
        self.obs_queue = deque(maxlen=3)
        initial_obs_shape = (self.num_envs, 
                             self._robot.data.root_lin_vel_b.shape[-1] 
                             + self._robot.data.root_ang_vel_b.shape[-1] + 1 +
                             3 + 3 + self.last_action.shape[-1])
        initial_obs = torch.zeros(initial_obs_shape, device=self.device)
        for _ in range(3):
            self.obs_queue.append(initial_obs)
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
        
        # Cylinder
        # self.cylinder_object = RigidObject(self.cfg.cylinder_cfg)
        # self.scene.rigid_objects["cylinder_0"] = self.cylinder_object
        # self.scene.rigid_objects["cylinder_1"] = self.cylinder_object
        # self.scene.rigid_objects["cylinder_2"] = self.cylinder_object
        
        # clone, filter, and replicate
        self.scene.clone_environments(copy_from_source=False)
        self.scene.filter_collisions(global_prim_paths=[self.cfg.terrain.prim_path])
        # add lights
        light_cfg = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.75, 0.75, 0.75))
        light_cfg.func("/World/Light", light_cfg)
        
        # self.dynamic_obstacle_cfg = RigidObjectCfg(
        #     prim_path="/World/envs/env_.*/dynamic_obstacle",
        #     spawn=sim_utils.CuboidCfg(
        #         size=(0.4, 0.8, 4),
        #         rigid_props=sim_utils.RigidBodyPropertiesCfg(max_depenetration_velocity=1.0, disable_gravity=True),
        #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
        #         physics_material=sim_utils.RigidBodyMaterialCfg(),
        #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1, 1.0, 1.0)),
        #         collision_props=sim_utils.CollisionPropertiesCfg(),  # <--- 添加这一行
        #     ),
        #     init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, 2)),
        # )
        # self.dynamic_obstacle = RigidObject(self.dynamic_obstacle_cfg)
        # self.scene.rigid_objects["dynamic_obstacle"] = self.dynamic_obstacle
        # ...已有代码...
        

    def _pre_physics_step(self, actions: torch.Tensor):     
        # vel =self._robot.data.root_lin_vel_w
        # pos =self._robot.data.root_pos_w
        # 让 dynamic_obstacle 在 x 轴做周期运动
        # t = self.episode_length_buf.float() * self.dt  # [num_envs]
        # new_pos = torch.zeros(self.num_envs, 3, device=self.device)
        # new_pos[:, 0] = 0.0 
        # new_pos[:, 1] = 2.0 * torch.sin(t)
        # new_pos[:, 2] = 2.0

        # new_rot = torch.zeros(self.num_envs, 4, device=self.device)
        # new_rot[:, 0] = 1.0

        # env_origins = self._terrain.terrain_origins.view(-1, 3).to(self.device)  # [num_envs, 3]
        # new_pos[:, :2] += env_origins[:, :2]  # 直接加，不用 env_ids

        # new_pose = torch.cat([new_pos, new_rot], dim=1)
        # self.dynamic_obstacle.write_root_pose_to_sim(new_pose)
        # # height = pos[..., 2]
        # velocity_z = vel[..., 2]
        # pos_error = 2 - height
        # target_acc = (
        #     4* pos_error
        #     + 2 * -velocity_z
        #     + 0.72
        # )
        # self.target_thrust = target_acc.unsqueeze(1) 
        # self.target_rate=torch.zeros_like(actions[:, 0:3])
        
        # push_ids=(self.episode_length_buf>300 ) & (self.episode_length_buf<350)
        # print("push_ids", push_ids)
        # self.target_rate[:, 0] = 0.3*torch.sin(self.episode_length_buf.float()/20)
        # self.target_rate[push_ids, 1] = -0.3
        # print("target_rate", self.target_rate)
        # print("target_thrust", self.target_thrust)
        
        
        # self.target_rate = torch.zeros_like(actions[:, 0:3])
        # self.target_thrust =torch.ones_like(actions[:, 3]).unsqueeze(1)

        # return
        #  set controller target here
        self.target_rate = actions[:, 0:3].clip(-1,1) * torch.pi
        self.target_thrust = actions[:, 3].clip(0,1).unsqueeze(1)
        self.last_action = actions.clip(-1,1)
        # self._actions = actions.clone().clamp(-1.0, 1.0)
        # self._thrust[:, 0, 2] = self.cfg.thrust_to_weight * self._robot_weight * (self._actions[:, 0] + 1.0) / 2.0
        # self._thrust[:, 0, 2] = 0.0
        # self._thrust[:, 2, 2] = 0.0       
        # self._moment[:, 0, :] = self.cfg.moment_scale * self._actions[:, 1:]

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
        # 只取后4个竖直分辨率
        current_scan = current_scan[..., 1:3]  # [num_envs, 72, 4]
        N = current_scan.shape[1]  # 72

        # 计算 desired_p_b 对应的角度
        desired_angle = torch.atan2(desired_p_b[:, 1], desired_p_b[:, 0])
        desired_angle = torch.where(desired_angle < 0, desired_angle + 2 * torch.pi, desired_angle)
        
        # 有效mask: 只用后三个高度层判断无遮挡
        valid_mask = (current_scan[..., 1:] < 0.2).all(dim=-1)  # [num_envs, N]
        angles = torch.linspace(0, 2 * torch.pi, N, device=self.device, dtype=desired_p_b.dtype)
        angles = angles.unsqueeze(0).expand(desired_p_b.shape[0], N)  # [num_envs, N]

        # 计算每个方向与目标方向的夹角差
        angle_diff = torch.remainder(angles - desired_angle.unsqueeze(1) + torch.pi, 2 * torch.pi) - torch.pi  # [-pi, pi]

        # 权重带宽调大，平滑
        weights = torch.exp(-0.5 * (angle_diff / 1.2) ** 2)
        weights = weights * valid_mask.float()
        weights_sum = weights.sum(dim=1, keepdim=True) + 1e-4
        weights = weights / weights_sum

        dir_x = torch.cos(angles)
        dir_y = torch.sin(angles)
        avg_dir = torch.stack([
            (dir_x * weights).sum(dim=1),
            (dir_y * weights).sum(dim=1)
        ], dim=-1)  # [num_envs, 2]

        # --- 关键优化：无遮挡方向极少时，直接选最近无遮挡方向 ---
        valid_counts = valid_mask.sum(dim=1)
        min_valid = 3  # 少于2个无遮挡方向时直接选最近无遮挡方向
        fallback_dir = torch.stack([dir_x, dir_y], dim=-1)
        # 对每个环境，找到最近的无遮挡方向
        masked_angle_diff = torch.abs(angle_diff) + (~valid_mask) * 1e6
        nearest_idx = masked_angle_diff.argmin(dim=1)
        nearest_dir = fallback_dir[torch.arange(desired_p_b.shape[0]), nearest_idx]
        # 平滑选择
        avg_dir = torch.where(
            (valid_counts > min_valid).unsqueeze(-1),
            avg_dir,
            nearest_dir
        )

        # 保持z方向
        optimal_direction = desired_p_b.clone()
        optimal_direction[:, :2] = avg_dir
        optimal_direction = optimal_direction / (optimal_direction.norm(dim=-1, keepdim=True) + 1e-6)
        return optimal_direction
        
    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        # 计算单位方向向量
        desired_pos_b  = desired_pos_b / torch.norm(desired_pos_b, dim=-1, keepdim=True)
        
        # desired_pos_b = desired_pos_b / 10.0 
        # obs = torch.cat(
        #     [
        #         self.add_guass_noise(self._robot.data.root_lin_vel_b,
        #                              self.cfg.noise_scales["root_lin_vel_b"]),
        #         self.add_guass_noise(self._robot.data.root_ang_vel_b,
        #                              self.cfg.noise_scales["root_ang_vel_b"]),
        #         self.add_guass_noise(self._robot.data.projected_gravity_b,
        #                              self.cfg.noise_scales["projected_gravity_b"]),
        #         desired_pos_b,
        #     ],
        #     dim=-1,
        # )
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
        
        # 更新队列
        # self.obs_queue.append(current_non_lidar_obs)
        
        # 累积最近3帧的非激光雷达部分观测数据
        # accumulated_non_lidar_obs = torch.cat(list(self.obs_queue), dim=-1)
        
        # 将激光雷达部分与累积的非激光雷达部分观测数据拼接
        obs = torch.cat(
            [
                self.current_scan_noise,
                # accumulated_non_lidar_obs,
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
                # self._robot.data.root_state_w[:, :3]/5,
                # self._robot.data.root_state_w[:, 3:7],
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
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 5)
        action_diff = torch.sum(torch.square(self.last_action - self._previous_actions), dim=1)
        
        # ----------------------------------------------------------------
        # get optimal direction
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
        # 计算单位方向向量
        desired_pos_b  = desired_pos_b / torch.norm(desired_pos_b, dim=-1, keepdim=True)
        self.optimal_direction = self.get_optimal_direction(desired_pos_b)
        # ---------------------------------------------------------------
        # 如果距离终点较近（例如小于0.8米），则 optimal_direction 直接等于 desired_pos_b
        close_to_goal = distance_to_goal < 0.6
        self.optimal_direction = torch.where(
            close_to_goal.unsqueeze(-1),
            desired_pos_b,
            self.optimal_direction
        )
        # vel reward
        vel_direction = (self._desired_pos_w - self._robot.data.root_pos_w)
        vel_direction = vel_direction / torch.norm(vel_direction, dim=-1, keepdim=True)
        # reward_dir = (self._robot.data.root_lin_vel_w * vel_direction).sum(-1).clip(max=8.0)
        reward_dir = (self._robot.data.root_lin_vel_b * self.optimal_direction).sum(-1).clip(max=8.0)
        reward_z = torch.exp(-5 * torch.abs(self._robot.data.root_pos_w[:, 2] - self._desired_pos_w[:, 2]))
        
        g_proj=self._robot.data.projected_gravity_b
        g_proj=g_proj/torch.linalg.norm(g_proj, dim=1, keepdim=True)
        # Reward for keeping the drone stable (aligned with gravity)
        g_proj_reward = torch.exp(-5 * torch.abs(-1 - g_proj[:, 2]))
        
        # reward_esdf = torch.exp(-5 * self.current_scan.max(dim=1).values)
        # 新增：偏航角奖励
        # 计算无人机当前yaw（世界坐标系下）
        quat_w = self._robot.data.root_quat_w
        # 提取yaw分量
        yaw = torch.atan2(
            2.0 * (quat_w[:, 0] * quat_w[:, 3] + quat_w[:, 1] * quat_w[:, 2]),
            1.0 - 2.0 * (quat_w[:, 2] ** 2 + quat_w[:, 3] ** 2)
        )
        # 目标方向在世界坐标系下的yaw
        target_vec = self._desired_pos_w - self._robot.data.root_pos_w
        target_yaw = torch.atan2(target_vec[:, 1], target_vec[:, 0])
        # 偏航角差 [-pi, pi]
        yaw_diff = torch.remainder(yaw - target_yaw + torch.pi, 2 * torch.pi) - torch.pi
        # 奖励偏航角对齐（高斯型或负绝对值）
        reward_yaw = torch.exp(-2 * torch.abs(yaw_diff))  # 你也可以用 -torch.abs(yaw_diff)
   
        # lidar 
        live = torch.ones_like(lin_vel)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "z": reward_z * self.cfg.z_reward_scale * self.step_dt,
            # "esdf": reward_esdf * self.cfg.esdf_scale * self.step_dt,
            # "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_diff" : action_diff* self.cfg.action_diff_reward_scale * self.step_dt,
            "live" : self.cfg.live_scale * live * self.step_dt,
            "reward_dir": reward_dir * self.cfg.dir_reward_scale * self.step_dt,
            "reward_g_proj": g_proj_reward * self.cfg.g_proj_reward_scale * self.step_dt,
            "reward_yaw": reward_yaw * self.cfg.yaw_reward_scale * self.step_dt,
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
        
        velocity_died = velocity_magnitude > 8.0
        
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
        # 重置障碍物到各自环境原点上方2米
        # ----------- 这里批量设置每个环境的障碍物初始位置 -----------
        # env_ids 是你要 reset 的环境索引
        # env_origins = self._terrain.terrain_origins.view(-1, 3).to(self.device)  # [num_envs, 3]

        # # 只对需要 reset 的 env_ids 进行操作
        # obstacle_pos = torch.zeros(len(env_ids), 3, device=self.device)
        # obstacle_pos[:, :2] += env_origins[env_ids, :2]  # 加上每个环境的原点xy
        # obstacle_pos[:, 2] = env_origins[env_ids, 2] + 2.0  # z轴加2米

        # obstacle_rot = torch.zeros(len(env_ids), 4, device=self.device)
        # obstacle_rot[:, 0] = 1.0

        # obstacle_pose = torch.cat([obstacle_pos, obstacle_rot], dim=1)  # [len(env_ids), 7]
        # self.dynamic_obstacle.write_root_pose_to_sim(obstacle_pose, env_ids=env_ids)
            
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
        random_yaw = torch.rand(len(env_ids), device=self.device) * 2 * torch.pi
        random_quaternions = torch.stack([
            torch.cos(random_yaw / 2),
            torch.zeros_like(random_yaw),
            torch.zeros_like(random_yaw),
            torch.sin(random_yaw / 2)
        ], dim=-1)
        default_root_state[:, 3:7] = random_quaternions
        
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
                marker_cfg.markers["arrow"].scale = (3.0, 0.3, 0.3)  # 设置箭头的大小
                # -- direction pose
                marker_cfg.prim_path = "/Visuals/Command/direction"
                self.direction_visualizer = VisualizationMarkers(marker_cfg)
             # set their visibility to true
            self.direction_visualizer.set_visibility(True)
            
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