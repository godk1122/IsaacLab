# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
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


class QuadcopterEnvWindow(BaseEnvWindow):
    """Window manager for the Quadcopter environment."""

    def __init__(self, env, window_name: str = "IsaacLab"):
        """Initialize the window.

        Args:
            env: The environment object.
            window_name: The name of the window. Defaults to "IsaacLab".
        """
        # initialize base window
        super().__init__(env, window_name)
        # add custom UI elements
        with self.ui_window_elements["main_vstack"]:
            with self.ui_window_elements["debug_frame"]:
                with self.ui_window_elements["debug_vstack"]:
                    # add command manager visualization
                    self._create_debug_vis_ui_element("targets", self.env)

 ##
# Pre-defined configs
##
# from isaaclab_assets import IRIS_CFG  # isort: skip
from isaaclab_assets import UAVLIDAR_CFG  # isort: skip
# from isaaclab_assets import UAVZXW_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip


@configclass
class QuadcopterEnvCfg(DirectRLEnvCfg):
    # env
    episode_length_s = 10
    decimation = 5
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = QuadcopterEnvWindow

    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=0.004,
        render_interval=decimation,
        # disable_contact_processing=False,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
    )
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="plane",
        collision_group=-1,
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        debug_vis=False,
    )

    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=4096, env_spacing=10, replicate_physics=True)

    # robot
    thrust_to_weight = 1.9
    moment_scale = 0.01

    # reward scales
    lin_vel_reward_scale = -0.05
    ang_vel_reward_scale = -0.05
    action_diff_reward_scale = -0.1
    distance_to_goal_reward_scale = 20.0
    rotor_speed_discount = 50
    
    robot: ArticulationCfg = UAVLIDAR_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            rot=(0.0, 0.0, 0.0, 1.0),
            pos=(0.0, 0.0, 0.5),
        )
    )
    class domain_randomization:
        class motor:
            enable = True
            class scale:
                force_constants = 0.2
                max_rotation_velocities = 0.2
                moment_constants = 0.2
                tau_up = 0.1
                tau_down = 0.1
        class noise:
            enable = False
            root_lin_vel_b = 0.0
            root_ang_vel_b = 0.0
            projected_gravity_b = 0.0    
            


@configclass
class TrackEnvCfg(QuadcopterEnvCfg):
    distance_to_goal_reward_scale = 1.0