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
from isaaclab.sensors import RayCaster, RayCasterCfg, patterns

from isaaclab.terrains import TerrainImporterCfg, TerrainImporter, TerrainGeneratorCfg, HfDiscreteObstaclesTerrainCfg, HfWallTerrainCfg
from isaaclab.terrains import HfDiscreteObstaclesWallTerrainCfg

class LidarFlyEnvWindow(BaseEnvWindow):
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
from isaaclab_assets import UAVLIDAR_CFG  # isort: skip
from isaaclab.markers import CUBOID_MARKER_CFG  # isort: skip
from isaaclab.terrains.config.rough import ROUGH_TERRAINS_CFG  # isort: skip

@configclass
class LidarFlyEnvCfg(DirectRLEnvCfg):
    # lidar
    lidar_vfov = (0.0, 20.0)
    lidar_resolution = (72,5)
    lidar_range = 10.0
    # env
    episode_length_s = 20
    decimation = 5
    action_space = 4
    observation_space = 12
    state_space = 0
    debug_vis = True

    ui_window_class_type = LidarFlyEnvWindow

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
        physx = sim_utils.PhysxCfg(
            gpu_found_lost_pairs_capacity = 108673297,
            gpu_total_aggregate_pairs_capacity = 8390656,
            gpu_found_lost_aggregate_pairs_capacity = 109045760,
        ),
    )
    
    # -------------------------------- terrain --------------------------------
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     terrain_generator=TerrainGeneratorCfg(
    #         size=(16.0, 10.0),
    #         border_width=1.0,
    #         # num_rows=32,
    #         # num_cols=32,
    #         num_rows=4,
    #         num_cols=4,
    #         horizontal_scale=0.1,
    #         vertical_scale=0.01,
    #         slope_threshold=0.75,
    #         use_cache=False,
    #         sub_terrains={
    #             "obstacles0": HfDiscreteObstaclesTerrainCfg(
    #                 # size=(4.0, 4.0),
    #                 horizontal_scale=0.1,
    #                 vertical_scale=0.01,
    #                 border_width=1.0,
    #                 num_obstacles= 14,
    #                 obstacle_height_mode="fixed",
    #                 obstacle_width_range=(0.4, 0.8),
    #                 obstacle_height_range=(4.0, 5.0),
    #                 platform_width= 0.0,
    #                 proportion=0.3,
    #             ),
    #             "obstacles1": HfDiscreteObstaclesTerrainCfg(
    #                 # size=(4.0, 4.0),
    #                 horizontal_scale=0.1,
    #                 vertical_scale=0.01,
    #                 border_width=1.0,
    #                 num_obstacles= 10,
    #                 obstacle_height_mode="fixed",
    #                 obstacle_width_range=(0.4, 0.8),
    #                 obstacle_height_range=(4.0, 5.0),
    #                 platform_width= 0.0,
    #                 proportion=0.2,
    #             ),
    #             "obstacles2": HfDiscreteObstaclesTerrainCfg(
    #                 # size=(4.0, 4.0),
    #                 horizontal_scale=0.1,
    #                 vertical_scale=0.01,
    #                 border_width=1.0,
    #                 num_obstacles= 8,
    #                 obstacle_height_mode="fixed",
    #                 obstacle_width_range=(0.4, 0.8),
    #                 obstacle_height_range=(4.0, 5.0),
    #                 platform_width= 0.0,
    #                 proportion=0.2,
    #             ),
    #             "obstacles2": HfDiscreteObstaclesTerrainCfg(
    #                 # size=(4.0, 4.0),
    #                 horizontal_scale=0.1,
    #                 vertical_scale=0.01,
    #                 border_width=1.0,
    #                 num_obstacles= 0,
    #                 obstacle_height_mode="fixed",
    #                 obstacle_width_range=(0.4, 0.8),
    #                 obstacle_height_range=(4.0, 5.0),
    #                 platform_width= 0.0,
    #                 proportion=0.3,
    #             ),
    #             # "wall1": HfWallTerrainCfg(
    #             #     horizontal_scale=0.1,
    #             #     vertical_scale=0.01,
    #             #     border_width=1.0,
    #             #     num_walls=1,
    #             #     wall_height_range=(4.0, 5.0),
    #             #     wall_width_range=(0.4, 0.6),
    #             #     wall_length_range=(4.0, 6.0),
    #             #     platform_width=0.0,
    #             #     proportion=0.2,
    #             # ),
    #             # "wall2": HfWallTerrainCfg(
    #             #     horizontal_scale=0.1,
    #             #     vertical_scale=0.01,
    #             #     border_width=1.0,
    #             #     num_walls=2,
    #             #     wall_height_range=(4.0, 5.0),
    #             #     wall_width_range=(0.4, 0.6),
    #             #     wall_length_range=(4.0, 6.0),
    #             #     platform_width=0.0,
    #             #     proportion=0.2,
    #             # ),
    #             # "wall3": HfWallTerrainCfg(
    #             #     horizontal_scale=0.1,
    #             #     vertical_scale=0.01,
    #             #     border_width=1.0,
    #             #     num_walls=3,
    #             #     wall_height_range=(4.0, 5.0),
    #             #     wall_width_range=(0.4, 0.6),
    #             #     wall_length_range=(4.0, 6.0),
    #             #     platform_width=0.0,
    #             #     proportion=0.1,
    #             # ),
    #         },
    #     ),
    #     collision_group=-1,
    #     debug_vis=True,
    # )
     # -------------------------- discrete_obstacles_wall_terrain --------------------------
    terrain = TerrainImporterCfg(
        prim_path="/World/ground",
        terrain_type="generator",
        physics_material=sim_utils.RigidBodyMaterialCfg(
            friction_combine_mode="multiply",
            restitution_combine_mode="multiply",
            static_friction=1.0,
            dynamic_friction=1.0,
            restitution=0.0,
        ),
        terrain_generator=TerrainGeneratorCfg(
            size=(16.0, 10.0),
            border_width=1.0,
            # num_rows=32,
            # num_cols=32,
            num_rows=4,
            num_cols=4,
            horizontal_scale=0.1,
            vertical_scale=0.005,
            slope_threshold=0.75,
            use_cache=False,
            sub_terrains={
                # most complexity terrain, mixed obstacles with walls
                "mixed_obstacles_1": HfDiscreteObstaclesWallTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.01,
                    border_width=1.0,
                    num_obstacles=14,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(4.0, 5.0),
                    num_cylinders=3,
                    cylinder_radius_range=(0.2, 0.6),
                    cylinder_height=4.5,
                    num_walls=1,
                    wall_height_range=(4.0, 5.0),
                    wall_width_range=(0.4, 0.6),
                    wall_length_range=(4.0, 6.0),
                    platform_width=0.0,
                    proportion=0.2,
                ),
                # middle complexity terrain, mixed obstacles with walls
                "mixed_obstacles_2": HfDiscreteObstaclesWallTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.01,
                    border_width=1.0,
                    num_obstacles=10,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(4.0, 5.0),
                    num_cylinders=2,
                    cylinder_radius_range=(0.2, 0.6),
                    cylinder_height=4.5,
                    num_walls=1,
                    wall_height_range=(4.0, 5.0),
                    wall_width_range=(0.4, 0.6),
                    wall_length_range=(4.0, 6.0),
                    platform_width=0.0,
                    proportion=0.3,
                ),
                # low complexity terrain, only cuboid obstacles
                "mixed_obstacles_3": HfDiscreteObstaclesWallTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.01,
                    border_width=1.0,
                    num_obstacles=6,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(4.0, 5.0),
                    num_cylinders=1,
                    cylinder_radius_range=(0.2, 0.6),
                    cylinder_height=4.5,
                    num_walls=1,
                    wall_height_range=(4.0, 5.0),
                    wall_width_range=(0.4, 0.6),
                    wall_length_range=(4.0, 6.0),
                    platform_width=0.0,
                    proportion=0.2,
                ),
                # only cuboid obstacles and cylinders
                "mixed_obstacles_4": HfDiscreteObstaclesWallTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.01,
                    border_width=1.0,
                    num_obstacles=15,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(4.0, 5.0),
                    num_cylinders=3,
                    cylinder_radius_range=(0.2, 0.6),
                    cylinder_height=4.5,
                    num_walls=0,
                    wall_height_range=(4.0, 5.0),
                    wall_width_range=(0.4, 0.6),
                    wall_length_range=(4.0, 6.0),
                    platform_width=0.0,
                    proportion=0.1,
                ),
                # only wall
                "mixed_obstacles_5": HfDiscreteObstaclesWallTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.01,
                    border_width=1.0,
                    num_obstacles=0,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(4.0, 5.0),
                    num_cylinders=0,
                    cylinder_radius_range=(0.2, 0.6),
                    cylinder_height=4.5,
                    num_walls=2,
                    wall_height_range=(4.0, 5.0),
                    wall_width_range=(0.4, 0.6),
                    wall_length_range=(4.0, 6.0),
                    platform_width=0.0,
                    proportion=0.1,
                ),
                # empty env
                "mixed_obstacles_6": HfDiscreteObstaclesWallTerrainCfg(
                    horizontal_scale=0.1,
                    vertical_scale=0.01,
                    border_width=1.0,
                    num_obstacles=0,
                    obstacle_height_mode="fixed",
                    obstacle_width_range=(0.4, 0.8),
                    obstacle_height_range=(4.0, 5.0),
                    num_cylinders=0,
                    cylinder_radius_range=(0.2, 0.6),
                    cylinder_height=4.5,
                    num_walls=0,
                    wall_height_range=(4.0, 5.0),
                    wall_width_range=(0.4, 0.6),
                    wall_length_range=(4.0, 6.0),
                    platform_width=0.0,
                    proportion=0.1,
                ),
            },
        ),
        collision_group=-1,
        debug_vis=True,
    )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="plane",
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )
    # terrain = TerrainImporterCfg(
    #     prim_path="/World/ground",
    #     terrain_type="generator",
    #     terrain_generator=ROUGH_TERRAINS_CFG,
    #     collision_group=-1,
    #     physics_material=sim_utils.RigidBodyMaterialCfg(
    #         friction_combine_mode="multiply",
    #         restitution_combine_mode="multiply",
    #         static_friction=1.0,
    #         dynamic_friction=1.0,
    #         restitution=0.0,
    #     ),
    #     debug_vis=False,
    # )
    # scene
    # scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=1024, env_spacing=2, replicate_physics=True)
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=16, env_spacing=2, replicate_physics=True)
    # robot
    robot: ArticulationCfg = UAVLIDAR_CFG.replace(
        prim_path="/World/envs/env_.*/Robot",
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(-8.5, 0.0, 1),
        )
    )
    # cylinder
    # Rigid Object
    # cylinder_cfg = RigidObjectCfg(
    #     prim_path="/World/envs/env_.*/Cylinder",
    #     spawn=sim_utils.CylinderCfg(
    #         radius=0.2,
    #         height=4,
    #         rigid_props=sim_utils.RigidBodyPropertiesCfg(),
    #         mass_props=sim_utils.MassPropertiesCfg(mass=1.0),
    #         collision_props=sim_utils.CollisionPropertiesCfg(),
    #         visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 1.0, 1.0), metallic=0.2),
    #     ),
    #     init_state=RigidObjectCfg.InitialStateCfg(
    #         pos= (0.0, 0.0, 2.0),
    #     ),
    # )
    
    # # lidar
    _lidar_vfov = (
        max(-89., lidar_vfov[0]),
        min(89., lidar_vfov[1])
    )
    ray_scanner = RayCasterCfg(
        prim_path="/World/envs/env_.*/Robot/base_link",
        update_period=0.1,
        offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.1)),
        attach_yaw_only=False,
        pattern_cfg=patterns.BpearlPatternCfg(
            vertical_ray_angles=torch.linspace(*_lidar_vfov, 5).tolist(),
            horizontal_res= 360/lidar_resolution[0],
        ),
        debug_vis=False,
        mesh_prim_paths=["/World/ground"],
    )

    thrust_to_weight = 1.9
    moment_scale = 0.01
    rotor_speed_discount = 50

    # reward scales
    ang_vel_reward_scale = -0.05
    z_reward_scale = 1.0 
    action_diff_reward_scale = -0.1
    live_scale = 1
    dir_reward_scale = 1.0
    yaw_reward_scale = 1.0
    distance_reward_scale = 0.1

    class domain_randomization:
        class motor:
            enable = True
            class scale:
                force_constants = 0.2
                max_rotation_velocities = 0.1
                moment_constants = 0.2
                tau_up = 0.2
                tau_down = 0.2
        
        class noise:
            enable = False
            root_lin_vel_b = 0.0
            root_ang_vel_b = 0.0
            projected_gravity_b = 0.0    
        
        class lidar_noise:
            enable = True
            noise = 0.03