# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Functions to generate height fields for different terrains."""

from __future__ import annotations

import numpy as np
import scipy.interpolate as interpolate
from typing import TYPE_CHECKING

from .utils import height_field_to_mesh

if TYPE_CHECKING:
    from . import hf_terrains_cfg


@height_field_to_mesh
def random_uniform_terrain(difficulty: float, cfg: hf_terrains_cfg.HfRandomUniformTerrainCfg) -> np.ndarray:
    """Generate a terrain with height sampled uniformly from a specified range.

    .. image:: ../../_static/terrains/height_field/random_uniform_terrain.jpg
       :width: 40%
       :align: center

    Note:
        The :obj:`difficulty` parameter is ignored for this terrain.

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the downsampled scale is smaller than the horizontal scale.
    """
    # check parameters
    # -- horizontal scale
    if cfg.downsampled_scale is None:
        cfg.downsampled_scale = cfg.horizontal_scale
    elif cfg.downsampled_scale < cfg.horizontal_scale:
        raise ValueError(
            "Downsampled scale must be larger than or equal to the horizontal scale:"
            f" {cfg.downsampled_scale} < {cfg.horizontal_scale}."
        )

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- downsampled scale
    width_downsampled = int(cfg.size[0] / cfg.downsampled_scale)
    length_downsampled = int(cfg.size[1] / cfg.downsampled_scale)
    # -- height
    height_min = int(cfg.noise_range[0] / cfg.vertical_scale)
    height_max = int(cfg.noise_range[1] / cfg.vertical_scale)
    height_step = int(cfg.noise_step / cfg.vertical_scale)

    # create range of heights possible
    height_range = np.arange(height_min, height_max + height_step, height_step)
    # sample heights randomly from the range along a grid
    height_field_downsampled = np.random.choice(height_range, size=(width_downsampled, length_downsampled))
    # create interpolation function for the sampled heights
    x = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_downsampled)
    y = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_downsampled)
    func = interpolate.RectBivariateSpline(x, y, height_field_downsampled)

    # interpolate the sampled heights to obtain the height field
    x_upsampled = np.linspace(0, cfg.size[0] * cfg.horizontal_scale, width_pixels)
    y_upsampled = np.linspace(0, cfg.size[1] * cfg.horizontal_scale, length_pixels)
    z_upsampled = func(x_upsampled, y_upsampled)
    # round off the interpolated heights to the nearest vertical step
    return np.rint(z_upsampled).astype(np.int16)


@height_field_to_mesh
def pyramid_sloped_terrain(difficulty: float, cfg: hf_terrains_cfg.HfPyramidSlopedTerrainCfg) -> np.ndarray:
    """Generate a terrain with a truncated pyramid structure.

    The terrain is a pyramid-shaped sloped surface with a slope of :obj:`slope` that trims into a flat platform
    at the center. The slope is defined as the ratio of the height change along the x axis to the width along the
    x axis. For example, a slope of 1.0 means that the height changes by 1 unit for every 1 unit of width.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_sloped_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_sloped_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    if cfg.inverted:
        slope = -cfg.slope_range[0] - difficulty * (cfg.slope_range[1] - cfg.slope_range[0])
    else:
        slope = cfg.slope_range[0] + difficulty * (cfg.slope_range[1] - cfg.slope_range[0])

    # switch parameters to discrete units
    # -- horizontal scale
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- height
    # we want the height to be 1/2 of the width since the terrain is a pyramid
    height_max = int(slope * cfg.size[0] / 2 / cfg.vertical_scale)
    # -- center of the terrain
    center_x = int(width_pixels / 2)
    center_y = int(length_pixels / 2)

    # create a meshgrid of the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    # offset the meshgrid to the center of the terrain
    xx = (center_x - np.abs(center_x - xx)) / center_x
    yy = (center_y - np.abs(center_y - yy)) / center_y
    # reshape the meshgrid to be 2D
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)
    # create a sloped surface
    hf_raw = np.zeros((width_pixels, length_pixels))
    hf_raw = height_max * xx * yy

    # create a flat platform at the center of the terrain
    platform_width = int(cfg.platform_width / cfg.horizontal_scale / 2)
    # get the height of the platform at the corner of the platform
    x_pf = width_pixels // 2 - platform_width
    y_pf = length_pixels // 2 - platform_width
    z_pf = hf_raw[x_pf, y_pf]
    hf_raw = np.clip(hf_raw, min(0, z_pf), max(0, z_pf))

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def pyramid_stairs_terrain(difficulty: float, cfg: hf_terrains_cfg.HfPyramidStairsTerrainCfg) -> np.ndarray:
    """Generate a terrain with a pyramid stair pattern.

    The terrain is a pyramid stair pattern which trims to a flat platform at the center of the terrain.

    If the :obj:`cfg.inverted` flag is set to :obj:`True`, the terrain is inverted such that
    the platform is at the bottom.

    .. image:: ../../_static/terrains/height_field/pyramid_stairs_terrain.jpg
       :width: 40%

    .. image:: ../../_static/terrains/height_field/inverted_pyramid_stairs_terrain.jpg
       :width: 40%

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    step_height = cfg.step_height_range[0] + difficulty * (cfg.step_height_range[1] - cfg.step_height_range[0])
    if cfg.inverted:
        step_height *= -1
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stairs
    step_width = int(cfg.step_width / cfg.horizontal_scale)
    step_height = int(step_height / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the steps
    current_step_height = 0
    start_x, start_y = 0, 0
    stop_x, stop_y = width_pixels, length_pixels
    while (stop_x - start_x) > platform_width and (stop_y - start_y) > platform_width:
        # increment position
        # -- x
        start_x += step_width
        stop_x -= step_width
        # -- y
        start_y += step_width
        stop_y -= step_width
        # increment height
        current_step_height += step_height
        # add the step
        hf_raw[start_x:stop_x, start_y:stop_y] = current_step_height

    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def discrete_obstacles_terrain(difficulty: float, cfg: hf_terrains_cfg.HfDiscreteObstaclesTerrainCfg) -> np.ndarray:
    """Generate a terrain with randomly generated obstacles as pillars with positive and negative heights.

    The terrain is a flat platform at the center of the terrain with randomly generated obstacles as pillars
    with positive and negative height. The obstacles are randomly generated cuboids with a random width and
    height. They are placed randomly on the terrain with a minimum distance of :obj:`cfg.platform_width`
    from the center of the terrain.

    .. image:: ../../_static/terrains/height_field/discrete_obstacles_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    obs_height = cfg.obstacle_height_range[0] + difficulty * (
        cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0]
    )

   # switch parameters to discrete units
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)

    # 计算x/y方向的像素范围（正方形区域，中心对齐）
    x_center = width_pixels // 2
    y_center = length_pixels // 2
    half_box = int(6 / cfg.horizontal_scale)
    x_min = max(0, x_center - half_box)
    x_max = min(width_pixels, x_center + half_box)
    y_min = max(0, y_center - half_box)
    y_max = min(length_pixels, y_center + half_box)

    min_dist = int(1.5 / cfg.horizontal_scale)  # 设定障碍物中心最小间距（如2米，可调整）

    hf_raw = np.zeros((width_pixels, length_pixels))
    num_obstacles = cfg.num_obstacles
    placed = 0
    centers = []

    while placed < num_obstacles:
        # sample size
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))
        if (x_max - x_min - width) <= 0 or (y_max - y_min - length) <= 0:
            continue  # 尺寸太大，跳过
        x_start = np.random.randint(x_min, x_max - width + 1)
        y_start = np.random.randint(y_min, y_max - length + 1)
        # 计算障碍物中心
        center = (x_start + width // 2, y_start + length // 2)
        # 检查与已放置障碍物的最小距离
        too_close = False
        for c in centers:
            if (abs(center[0] - c[0]) < min_dist) and (abs(center[1] - c[1]) < min_dist):
                too_close = True
                break
        if too_close:
            continue  # 距离太近，重新采样
        hf_raw[x_start : x_start + width, y_start : y_start + length] = height
        centers.append(center)
        placed += 1

    return np.rint(hf_raw).astype(np.int16)

@height_field_to_mesh
def discrete_obstacles_wall_terrain(difficulty: float, cfg: hf_terrains_cfg.HfDiscreteObstaclesWallTerrainCfg, custom_walls: list[dict] = None) -> np.ndarray:
    """
    生成包含离散圆柱、长方体和墙壁的地形
    """
    # 基础参数
    obs_height = cfg.obstacle_height_range[0] + difficulty * (cfg.obstacle_height_range[1] - cfg.obstacle_height_range[0])
    obs_height = int(obs_height / cfg.vertical_scale)
    obs_width_min = int(cfg.obstacle_width_range[0] / cfg.horizontal_scale)
    obs_width_max = int(cfg.obstacle_width_range[1] / cfg.horizontal_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    hf_raw = np.zeros((width_pixels, length_pixels))

    # 1. 随机生成长方体障碍物（带最小距离约束）
    num_obstacles = cfg.num_obstacles
    obs_width_range = np.arange(obs_width_min, obs_width_max, 4)
    obs_length_range = np.arange(obs_width_min, obs_width_max, 4)
    min_dist = int(1.5 / cfg.horizontal_scale)  # 设定障碍物中心最小间距（如2米，可调整）
    placed = 0
    cuboid_centers = []
    while placed < num_obstacles:
        if cfg.obstacle_height_mode == "choice":
            height = np.random.choice([-obs_height, -obs_height // 2, obs_height // 2, obs_height])
        elif cfg.obstacle_height_mode == "fixed":
            height = obs_height
        else:
            raise ValueError(f"Unknown obstacle height mode '{cfg.obstacle_height_mode}'. Must be 'choice' or 'fixed'.")
        width = int(np.random.choice(obs_width_range))
        length = int(np.random.choice(obs_length_range))
        if (width_pixels - width) <= 0 or (length_pixels - length) <= 0:
            continue  # 尺寸太大，跳过
        x_start = np.random.randint(0, width_pixels - width + 1)
        y_start = np.random.randint(0, length_pixels - length + 1)
        center = (x_start + width // 2, y_start + length // 2)
        too_close = False
        for c in cuboid_centers:
            if (abs(center[0] - c[0]) < min_dist) and (abs(center[1] - c[1]) < min_dist):
                too_close = True
                break
        if too_close:
            continue
        hf_raw[x_start:x_start + width, y_start:y_start + length] = height
        cuboid_centers.append(center)
        placed += 1

    # 2. 随机生成圆柱障碍物（带最小距离约束）
    num_cylinders = getattr(cfg, "num_cylinders", 0)
    cyl_radius_min = int(getattr(cfg, "cylinder_radius_range", [2, 6])[0] / cfg.horizontal_scale)
    cyl_radius_max = int(getattr(cfg, "cylinder_radius_range", [2, 6])[1] / cfg.horizontal_scale)
    cyl_height = int(getattr(cfg, "cylinder_height", obs_height) / cfg.vertical_scale)
    cylinder_centers = []
    min_cyl_dist = int(1.5 / cfg.horizontal_scale)  # 可单独设置
    placed_cyl = 0
    while placed_cyl < num_cylinders:
        radius = np.random.randint(cyl_radius_min, cyl_radius_max + 1)
        if (width_pixels - 2 * radius) <= 0 or (length_pixels - 2 * radius) <= 0:
            continue
        x_center = np.random.randint(radius, width_pixels - radius)
        y_center = np.random.randint(radius, length_pixels - radius)
        center = (x_center, y_center)
        too_close = False
        for c in cylinder_centers:
            if (abs(center[0] - c[0]) < min_cyl_dist) and (abs(center[1] - c[1]) < min_cyl_dist):
                too_close = True
                break
        if too_close:
            continue
        yy, xx = np.meshgrid(np.arange(width_pixels), np.arange(length_pixels), indexing='ij')
        mask = (yy - x_center) ** 2 + (xx - y_center) ** 2 <= radius ** 2
        hf_raw[mask] = cyl_height
        cylinder_centers.append(center)
        placed_cyl += 1

    # 3. 随机生成墙壁障碍物（参考 wall_terrain 逻辑）
    num_walls = getattr(cfg, "num_walls", 0)
    wall_height = cfg.wall_height_range[0] + difficulty * (cfg.wall_height_range[1] - cfg.wall_height_range[0])
    wall_height = int(wall_height / cfg.vertical_scale)
    wall_width_min = int(cfg.wall_width_range[0] / cfg.horizontal_scale)
    wall_width_max = int(cfg.wall_width_range[1] / cfg.horizontal_scale)
    wall_length_min = int(cfg.wall_length_range[0] / cfg.horizontal_scale)
    wall_length_max = int(cfg.wall_length_range[1] / cfg.horizontal_scale)
    for _ in range(num_walls):
        wall_width = int(np.random.randint(wall_width_min, wall_width_max + 1))
        wall_length = int(np.random.randint(wall_length_min, wall_length_max + 1))
        if wall_length < wall_width:
            wall_length, wall_width = wall_width, wall_length
        x_start = np.random.randint(0, width_pixels - wall_width)
        y_start = np.random.randint(0, length_pixels - wall_length)
        hf_raw[x_start:x_start + wall_width, y_start:y_start + wall_length] = wall_height

    # 5. 中心平台清空
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    
    return np.rint(hf_raw).astype(np.int16)

@height_field_to_mesh
def wave_terrain(difficulty: float, cfg: hf_terrains_cfg.HfWaveTerrainCfg) -> np.ndarray:
    r"""Generate a terrain with a wave pattern.

    The terrain is a flat platform at the center of the terrain with a wave pattern. The wave pattern
    is generated by adding sinusoidal waves based on the number of waves and the amplitude of the waves.

    The height of the terrain at a point :math:`(x, y)` is given by:

    .. math::

        h(x, y) =  A \left(\sin\left(\frac{2 \pi x}{\lambda}\right) + \cos\left(\frac{2 \pi y}{\lambda}\right) \right)

    where :math:`A` is the amplitude of the waves, :math:`\lambda` is the wavelength of the waves.

    .. image:: ../../_static/terrains/height_field/wave_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.

    Raises:
        ValueError: When the number of waves is non-positive.
    """
    # check number of waves
    if cfg.num_waves < 0:
        raise ValueError(f"Number of waves must be a positive integer. Got: {cfg.num_waves}.")

    # resolve terrain configuration
    amplitude = cfg.amplitude_range[0] + difficulty * (cfg.amplitude_range[1] - cfg.amplitude_range[0])
    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    amplitude_pixels = int(0.5 * amplitude / cfg.vertical_scale)

    # compute the wave number: nu = 2 * pi / lambda
    wave_length = length_pixels / cfg.num_waves
    wave_number = 2 * np.pi / wave_length
    # create meshgrid for the terrain
    x = np.arange(0, width_pixels)
    y = np.arange(0, length_pixels)
    xx, yy = np.meshgrid(x, y, sparse=True)
    xx = xx.reshape(width_pixels, 1)
    yy = yy.reshape(1, length_pixels)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))
    # add the waves
    hf_raw += amplitude_pixels * (np.cos(yy * wave_number) + np.sin(xx * wave_number))
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def stepping_stones_terrain(difficulty: float, cfg: hf_terrains_cfg.HfSteppingStonesTerrainCfg) -> np.ndarray:
    """Generate a terrain with a stepping stones pattern.

    The terrain is a stepping stones pattern which trims to a flat platform at the center of the terrain.

    .. image:: ../../_static/terrains/height_field/stepping_stones_terrain.jpg
       :width: 40%
       :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve terrain configuration
    stone_width = cfg.stone_width_range[1] - difficulty * (cfg.stone_width_range[1] - cfg.stone_width_range[0])
    stone_distance = cfg.stone_distance_range[0] + difficulty * (
        cfg.stone_distance_range[1] - cfg.stone_distance_range[0]
    )

    # switch parameters to discrete units
    # -- terrain
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    # -- stones
    stone_distance = int(stone_distance / cfg.horizontal_scale)
    stone_width = int(stone_width / cfg.horizontal_scale)
    stone_height_max = int(cfg.stone_height_max / cfg.vertical_scale)
    # -- holes
    holes_depth = int(cfg.holes_depth / cfg.vertical_scale)
    # -- platform
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)
    # create range of heights
    stone_height_range = np.arange(-stone_height_max - 1, stone_height_max, step=1)

    # create a terrain with a flat platform at the center
    hf_raw = np.full((width_pixels, length_pixels), holes_depth)
    # add the stones
    start_x, start_y = 0, 0
    # -- if the terrain is longer than it is wide then fill the terrain column by column
    if length_pixels >= width_pixels:
        while start_y < length_pixels:
            # ensure that stone stops along y-axis
            stop_y = min(length_pixels, start_y + stone_width)
            # randomly sample x-position
            start_x = np.random.randint(0, stone_width)
            stop_x = max(0, start_x - stone_distance)
            # fill first stone
            hf_raw[0:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
            # fill row with stones
            while start_x < width_pixels:
                stop_x = min(width_pixels, start_x + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_x += stone_width + stone_distance
            # update y-position
            start_y += stone_width + stone_distance
    elif width_pixels > length_pixels:
        while start_x < width_pixels:
            # ensure that stone stops along x-axis
            stop_x = min(width_pixels, start_x + stone_width)
            # randomly sample y-position
            start_y = np.random.randint(0, stone_width)
            stop_y = max(0, start_y - stone_distance)
            # fill first stone
            hf_raw[start_x:stop_x, 0:stop_y] = np.random.choice(stone_height_range)
            # fill column with stones
            while start_y < length_pixels:
                stop_y = min(length_pixels, start_y + stone_width)
                hf_raw[start_x:stop_x, start_y:stop_y] = np.random.choice(stone_height_range)
                start_y += stone_width + stone_distance
            # update x-position
            start_x += stone_width + stone_distance
    # add the platform in the center
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)


@height_field_to_mesh
def wall_terrain(difficulty: float, cfg: hf_terrains_cfg.HfWallTerrainCfg) -> np.ndarray:
    """Generate a terrain with several randomly placed walls.

    The terrain is a flat platform at the center, with several walls (rectangular obstacles) placed randomly.
    Each wall has a random length, width, and height, and its long edge is always perpendicular to the x-axis (i.e., walls are aligned along the y-axis).

    .. image:: ../../_static/terrains/height_field/wall_terrain.jpg
        :width: 40%
        :align: center

    Args:
        difficulty: The difficulty of the terrain. This is a value between 0 and 1.
        cfg: The configuration for the terrain.

    Returns:
        The height field of the terrain as a 2D numpy array with discretized heights.
        The shape of the array is (width, length), where width and length are the number of points
        along the x and y axis, respectively.
    """
    # resolve wall configuration
    wall_height = cfg.wall_height_range[0] + difficulty * (cfg.wall_height_range[1] - cfg.wall_height_range[0])
    wall_width_min = int(cfg.wall_width_range[0] / cfg.horizontal_scale)
    wall_width_max = int(cfg.wall_width_range[1] / cfg.horizontal_scale)
    wall_length_min = int(cfg.wall_length_range[0] / cfg.horizontal_scale)
    wall_length_max = int(cfg.wall_length_range[1] / cfg.horizontal_scale)
    wall_height = int(wall_height / cfg.vertical_scale)

    # switch parameters to discrete units
    width_pixels = int(cfg.size[0] / cfg.horizontal_scale)
    length_pixels = int(cfg.size[1] / cfg.horizontal_scale)
    platform_width = int(cfg.platform_width / cfg.horizontal_scale)

    # create a terrain with a flat platform at the center
    hf_raw = np.zeros((width_pixels, length_pixels))

    for _ in range(cfg.num_walls):
        # Always make the long edge perpendicular to x-axis (i.e., along y-axis)
        wall_width = int(np.random.randint(wall_width_min, wall_width_max + 1))
        wall_length = int(np.random.randint(wall_length_min, wall_length_max + 1))
        # Ensure wall_length >= wall_width
        if wall_length < wall_width:
            wall_length, wall_width = wall_width, wall_length
        # Randomly choose position
        x_start = np.random.randint(-4, width_pixels - wall_width - 5)
        y_start = np.random.randint(-5, length_pixels - wall_length)
        # Place wall
        hf_raw[x_start:x_start + wall_width, y_start:y_start + wall_length] = wall_height

    # clip the terrain to the platform
    x1 = (width_pixels - platform_width) // 2
    x2 = (width_pixels + platform_width) // 2
    y1 = (length_pixels - platform_width) // 2
    y2 = (length_pixels + platform_width) // 2
    hf_raw[x1:x2, y1:y2] = 0
    # round off the heights to the nearest vertical step
    return np.rint(hf_raw).astype(np.int16)