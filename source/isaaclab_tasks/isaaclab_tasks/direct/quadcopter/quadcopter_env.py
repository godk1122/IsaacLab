# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import gymnasium as gym
import torch

import isaaclab.sim as sim_utils
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import DirectRLEnv, DirectRLEnvCfg
from isaaclab.envs.ui import BaseEnvWindow
from isaaclab.markers import VisualizationMarkers
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sim import SimulationCfg
from isaaclab.terrains import TerrainImporterCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import subtract_frame_transforms

##
# Pre-defined configs
##
from isaaclab_assets import UAVLIDAR_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip

from isaaclab_tasks.direct.quadcopter.modules.controller import RateController
from isaaclab_tasks.direct.quadcopter.modules.motor import MotorModel

from .quadcopter_cfg import QuadcopterEnvCfg

class QuadcopterEnv(DirectRLEnv):
    cfg: QuadcopterEnvCfg

    def __init__(self, cfg: QuadcopterEnvCfg, render_mode: str | None = None, **kwargs):
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
        self._episode_sums = {
            key: torch.zeros(self.num_envs, dtype=torch.float, device=self.device)
            for key in [
                "lin_vel",
                "ang_vel",
                "distance_to_goal",
                "action_diff",
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
        # vel =self._robot.data.root_lin_vel_w
        # pos =self._robot.data.root_pos_w
        
        # height = pos[..., 2]
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
        self.last_action = actions
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

    def add_guass_noise(self,value,scale):
        # add a noise to the value from -scale to scale
        return value+torch.randn_like(value, device=self.device)*scale
        
    def _get_observations(self) -> dict:
        desired_pos_b, _ = subtract_frame_transforms(
            self._robot.data.root_state_w[:, :3], self._robot.data.root_state_w[:, 3:7], self._desired_pos_w
        )
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
        obs = torch.cat(
            [
                self._robot.data.root_lin_vel_b,
                self._robot.data.root_ang_vel_b,
                g_proj,
                desired_pos_b,
                self.last_action,
            ],
            dim=-1,
        )
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
            
        observations = {"policy": obs}
        return observations

    def _get_rewards(self) -> torch.Tensor:
        lin_vel = torch.sum(torch.square(self._robot.data.root_lin_vel_b), dim=1)
        ang_vel = torch.sum(torch.square(self._robot.data.root_ang_vel_b), dim=1)
        distance_to_goal = torch.linalg.norm(self._desired_pos_w - self._robot.data.root_pos_w, dim=1)
        distance_to_goal_mapped = 1 - torch.tanh(distance_to_goal / 4.0)
        action_diff = torch.sum(torch.square(self.last_action - self._previous_actions), dim=1)
        rewards = {
            "lin_vel": lin_vel * self.cfg.lin_vel_reward_scale * self.step_dt,
            "ang_vel": ang_vel * self.cfg.ang_vel_reward_scale * self.step_dt,
            "distance_to_goal": distance_to_goal_mapped * self.cfg.distance_to_goal_reward_scale * self.step_dt,
            "action_diff" : action_diff* self.cfg.action_diff_reward_scale * self.step_dt,
        }
        reward = torch.sum(torch.stack(list(rewards.values())), dim=0)
        # Logging
        for key, value in rewards.items():
            self._episode_sums[key] += value
        return reward

    def _get_dones(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        time_out = self.episode_length_buf >= self.max_episode_length - 1
        died = torch.logical_or(self._robot.data.root_pos_w[:, 2] < 0.25, self._robot.data.root_pos_w[:, 2] > 3)
        
        # 创建具体的终止条件tensor而不是None
        height_died = torch.zeros_like(died)  # 暂时设为全False
        lidar_died = torch.zeros_like(died)  # 暂时设为全False
        velocity_died = torch.zeros_like(died)  # 暂时设为全False
        
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
        # Sample new commands
        self._desired_pos_w[env_ids, :2] = torch.zeros_like(self._desired_pos_w[env_ids, :2]).uniform_(-2.0, 2.0)
        self._desired_pos_w[env_ids, :2] += self._terrain.env_origins[env_ids, :2]
        self._desired_pos_w[env_ids, 2] = torch.zeros_like(self._desired_pos_w[env_ids, 2]).uniform_(0.5, 1.5)
        # Reset robot state
        joint_pos = self._robot.data.default_joint_pos[env_ids]
        joint_vel = self._robot.data.default_joint_vel[env_ids]
        default_root_state = self._robot.data.default_root_state[env_ids]
        default_root_state[:, :3] += self._terrain.env_origins[env_ids]
        self._robot.write_root_pose_to_sim(default_root_state[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(default_root_state[:, 7:], env_ids)
        self._robot.write_joint_state_to_sim(joint_pos, joint_vel, None, env_ids)

    def _set_debug_vis_impl(self, debug_vis: bool):
        # create markers if necessary for the first tome
        if debug_vis:
            if not hasattr(self, "goal_pos_visualizer"):
                marker_cfg = CUBOID_MARKER_CFG.copy()
                marker_cfg.markers["cuboid"].size = (0.05, 0.05, 0.05)
                # -- goal pose
                marker_cfg.prim_path = "/Visuals/Command/goal_position"
                self.goal_pos_visualizer = VisualizationMarkers(marker_cfg)
            # set their visibility to true
            self.goal_pos_visualizer.set_visibility(True)
        else:
            if hasattr(self, "goal_pos_visualizer"):
                self.goal_pos_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        # update the markers
        self.goal_pos_visualizer.visualize(self._desired_pos_w)
