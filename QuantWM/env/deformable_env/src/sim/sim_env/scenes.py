import numpy as np
from scipy.spatial.transform import Rotation
from ..utils import rand_float, quaternion_multuply

"""
Support Scenes:
        1. rope_scene
        2. granular_scene
        3. cloth_scene
"""


def rope_scene(obj_params):
    radius = 0.03
    # rope position
    rope_trans = [0.0, 0.5, 2.0]  # [x, y, z]

    # rope scale (length and thickness)
    # rope_length = rand_float(2.5, 3.0)
    rope_length = obj_params["rope_length"]
    rope_thickness = 3.0
    rope_scale = np.array([rope_length, rope_thickness, rope_thickness]) * 50

    # rope stiffness
    # stiffness = np.random.rand()
    stiffness = obj_params["stiffness"]
    if stiffness < 0.5:
        global_stiffness = stiffness * 1e-4 / 0.5
        cluster_spacing = 2 + 8 * stiffness
    else:
        global_stiffness = (stiffness - 0.5) * 4e-4 + 1e-4
        cluster_spacing = 6 + 4 * (stiffness - 0.5)

    # rope frtction
    dynamicFriction = 0.1

    # rope rotation
    # z_rotation = rand_float(10, 20)
    z_rotation = obj_params["z_rotation"]
    y_rotation = 90.0
    rot_1 = Rotation.from_euler("xyz", [0, y_rotation, 0.0], degrees=True)
    rotate_1 = rot_1.as_quat()
    rot_2 = Rotation.from_euler("xyz", [0, 0, z_rotation], degrees=True)
    rotate_2 = rot_2.as_quat()
    rope_rotate = quaternion_multuply(rotate_1, rotate_2)

    # others (ususally fixed)
    cluster_radius = 0.0
    cluster_stiffness = 0.55

    link_radius = 0.0
    link_stiffness = 1.0

    surface_sampling = 0.0
    volume_sampling = 4.0

    skinning_falloff = 5.0
    skinning_max_dist = 100.0

    cluster_plastic_threshold = 0.0
    cluster_plastic_creep = 0.0

    particleFriction = 0.25

    draw_mesh = 1

    relaxtion_factor = 1.0
    collisionDistance = radius * 0.5

    # params
    scene_params = np.array(
        [
            *rope_scale,
            *rope_trans,
            radius,
            cluster_spacing,
            cluster_radius,
            cluster_stiffness,
            link_radius,
            link_stiffness,
            global_stiffness,
            surface_sampling,
            volume_sampling,
            skinning_falloff,
            skinning_max_dist,
            cluster_plastic_threshold,
            cluster_plastic_creep,
            dynamicFriction,
            particleFriction,
            draw_mesh,
            relaxtion_factor,
            *rope_rotate,
            collisionDistance,
        ]
    )

    property_params = {
        "particle_radius": radius,
        "length": rope_length,
        "thickness": rope_thickness,
        "dynamic_friction": dynamicFriction,
        "cluster_spacing": cluster_spacing,
        "global_stiffness": global_stiffness,
        "stiffness": stiffness,
    }

    return scene_params, property_params


def granular_scene(obj_params):
    radius = 0.03
    # print(1/0)

    # granular_scale = rand_float(0.1, 0.3)
    granular_scale = obj_params["granular_scale"]

    # area = rand_float(1**2, 3**2)
    # area = 1.5
    area = obj_params["area"]

    # xz_ratio = rand_float(0.8, 1.2)
    # xz_ratio = 0.8
    xz_ratio = obj_params["xz_ratio"]

    x_max = area**0.5 * 0.5 * xz_ratio**0.5
    x_min = -x_max
    z_max = area**0.5 * 0.5 * xz_ratio**-0.5
    z_min = -z_max

    # granular_dis = rand_float(0.1 * granular_scale, 0.2 * granular_scale)
    granular_dis = obj_params["granular_dis"]

    num_granular_ft_x = (x_max - x_min - granular_scale) / (
        granular_dis + granular_scale
    ) + 1
    num_granular_ft_z = (z_max - z_min - granular_scale) / (
        granular_dis + granular_scale
    ) + 1

    # shape
    shape_type = 0  # 0: irreular shape; 1: regular shape
    shape_min_dist = 5.0  # 5. for irregular shape; 8 for regulra shape
    shape_max_dist = 10.0

    num_granular_ft_y = 1
    num_granular_ft = [num_granular_ft_x, num_granular_ft_y, num_granular_ft_z]
    num_granular = int(num_granular_ft_x * num_granular_ft_y * num_granular_ft_z)

    print(f"NUM_GRANULAR: {num_granular}")

    pos_granular = [-1.0, 1.0, -1.0]

    draw_mesh = 1

    shapeCollisionMargin = 0.01
    collisionDistance = 0.03

    dynamic_friction = 1.0
    granular_mass = 0.05

    scene_params = np.array(
        [
            radius,
            *num_granular_ft,
            granular_scale,
            *pos_granular,
            granular_dis,
            draw_mesh,
            shapeCollisionMargin,
            collisionDistance,
            dynamic_friction,
            granular_mass,
            shape_type,
            shape_min_dist,
            shape_max_dist,
        ]
    )

    property_param = {
        "particle_radius": radius,
        "granular_scale": granular_scale,
        "num_granular": num_granular,
        "distribution_r": granular_dis,
        "dynamic_friction": dynamic_friction,
        "granular_mass": granular_mass,
        "area": area,
        "xz_ratio": xz_ratio,
    }

    return scene_params, property_param


def cloth_scene():
    particle_r = 0.03
    cloth_pos = [-0.5, 1.0, 0.0]
    cloth_size = np.array([1.0, 1.0]) * 70.0

    """
        stretch stiffness: resistance to lengthwise stretching
        bend stiffness: resistance to bending
        shear stiffness: resistance to forces that cause sliding or twisting deformation
        """
    sf = np.random.rand()
    stiffness_factor = sf * 1.4 + 0.1
    stiffness = np.array([1.0, 1.0, 1.0]) * stiffness_factor
    stiffness[0] = np.clip(stiffness[0], 1.0, 1.5)
    dynamicFriction = -sf * 0.9 + 1.0

    cloth_mass = 0.1

    render_mode = 2  # 1: particles; 2: mesh
    flip_mesh = 0

    staticFriction = 0.0
    particleFriction = 0.0

    scene_params = np.array(
        [
            *cloth_pos,
            *cloth_size,
            *stiffness,
            cloth_mass,
            particle_r,
            render_mode,
            flip_mesh,
            dynamicFriction,
            staticFriction,
            particleFriction,
        ]
    )

    property_params = {
        "particle_radius": particle_r,
        "stretch_stiffness": stiffness[0],
        "bend_stiffness": stiffness[1],
        "shear_stiffness": stiffness[2],
        "dynamic_friction": dynamicFriction,
        "sf": sf,
    }

    return scene_params, property_params
