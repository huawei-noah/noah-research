import os
import numpy as np
import pyflex
import gym
import math
from scipy.spatial.distance import cdist

import pybullet as p
import pybullet_data

from .robot_env import FlexRobotHelper

pyflex.loadURDF = FlexRobotHelper.loadURDF
pyflex.resetJointState = FlexRobotHelper.resetJointState
pyflex.getRobotShapeStates = FlexRobotHelper.getRobotShapeStates

from .flex_scene import FlexScene
from .cameras import Camera
from ..utils import fps_with_idx, quatFromAxisAngle, find_min_distance, rand_float

BASE_DIR = os.path.abspath(os.path.join(__file__, "../../../../../../"))

class FlexEnv(gym.Env):
    def __init__(self, config=None) -> None:
        super().__init__()

        self.dataset_config = config["dataset"]

        # env component
        self.obj = self.dataset_config["obj"]
        self.obj_params = self.dataset_config["obj_params"]
        self.scene = FlexScene()

        # set up pybullet
        physicsClient = p.connect(p.DIRECT)
        p.setAdditionalSearchPath(pybullet_data.getDataPath())
        p.setGravity(0, 0, -10)
        self.planeId = p.loadURDF("plane.urdf")

        # set up robot arm
        # xarm6
        self.flex_robot_helper = FlexRobotHelper()
        self.end_idx = self.dataset_config["robot_end_idx"]
        self.num_dofs = self.dataset_config["robot_num_dofs"]
        self.robot_speed_inv = self.dataset_config["robot_speed_inv"]

        # set up pyflex
        self.screenWidth = self.dataset_config["screenWidth"]
        self.screenHeight = self.dataset_config["screenHeight"]
        self.camera = Camera(self.screenWidth, self.screenHeight)

        pyflex.set_screenWidth(self.screenWidth)
        pyflex.set_screenHeight(self.screenHeight)
        pyflex.set_light_dir(np.array([0.1, 5.0, 0.1]))
        pyflex.set_light_fov(70.0)
        pyflex.init(self.dataset_config["headless"])

        # set up camera
        self.camera_view = self.dataset_config["camera_view"]

        # define action space
        self.action_dim = self.dataset_config["action_dim"]
        self.action_space = self.dataset_config["action_space"]

        # stat
        self.count = 0
        self.imgs_list = []
        self.particle_pos_list = []
        self.eef_states_list = []

        self.fps = self.dataset_config["fps"]
        self.fps_number = self.dataset_config["fps_number"]

        # others
        self.gripper = self.dataset_config["gripper"]
        self.stick_len = self.dataset_config["pusher_len"]
    
        self.scene.set_scene(self.obj, self.obj_params)
        # set camera
        self.camera.set_init_camera(self.camera_view)
        save_data = True # default to render obs
        if save_data:
            (
                self.camPos_list,
                self.camAngle_list,
                self.cam_intrinsic_params,
                self.cam_extrinsic_matrix,
            ) = self.camera.init_multiview_cameras()
        # add table
        self.add_table()
        ## add robot
        self.add_robot()

    ### shape states
    def robot_to_shape_states(self, robot_states):
        n_robot_links = robot_states.shape[0]
        n_table = self.table_shape_states.shape[0]

        shape_states = np.zeros((n_table + n_robot_links, 14))
        shape_states[:n_table] = self.table_shape_states  # set shape states for table
        shape_states[n_table:] = robot_states  # set shape states for robot

        return shape_states

    def reset_robot(self, jointPositions=np.zeros(13).tolist()):
        index = 0
        for j in range(7):
            p.changeDynamics(self.robotId, j, linearDamping=0, angularDamping=0)
            info = p.getJointInfo(self.robotId, j)

            jointType = info[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                pyflex.resetJointState(self.flex_robot_helper, j, jointPositions[index])
                index = index + 1

        pyflex.set_shape_states(
            self.robot_to_shape_states(
                pyflex.getRobotShapeStates(self.flex_robot_helper)
            )
        )

    def add_table(self):
        ## add table board
        self.table_shape_states = np.zeros((2, 14))
        # table for workspace
        self.wkspace_height = 0.5
        self.wkspace_width = 3.5  # 3.5*2=7 grid = 700mm
        self.wkspace_length = 4.5  # 4.5*2=9 grid = 900mm
        halfEdge = np.array(
            [self.wkspace_width, self.wkspace_height, self.wkspace_length]
        )
        center = np.array([0.0, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0.0, 1.0, 0.0]), angle=0.0)
        hideShape = 0
        color = np.ones(3) * (160.0 / 255.0)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states[0] = np.concatenate([center, center, quats, quats])

        # table for robot
        if self.obj in ["cloth"]:
            robot_table_height = 0.5 + 1.0
        else:
            robot_table_height = 0.5 + 0.3
        robot_table_width = 126 / 200  # 126mm
        robot_table_length = 126 / 200  # 126mm
        halfEdge = np.array([robot_table_width, robot_table_height, robot_table_length])
        center = np.array([-self.wkspace_width - robot_table_width, 0.0, 0.0])
        quats = quatFromAxisAngle(axis=np.array([0.0, 1.0, 0.0]), angle=0.0)
        hideShape = 0
        color = np.ones(3) * (160.0 / 255.0)
        pyflex.add_box(halfEdge, center, quats, hideShape, color)
        self.table_shape_states[1] = np.concatenate([center, center, quats, quats])

    def add_robot(self):
        if self.obj in ["granular"]:
            # flat board pusher
            robot_base_pos = [-self.wkspace_width - 0.6, 0.0, self.wkspace_height + 0.3]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(
                self.flex_robot_helper,
                os.path.join(BASE_DIR, "env/deformable_env/src/sim/assets/xarm/xarm6_with_gripper_board.urdf"),
                robot_base_pos,
                robot_base_orn,
                globalScaling=10.0,
            )
            self.rest_joints = np.zeros(8)
        elif self.obj in ["rope"]:
            # stick pusher
            robot_base_pos = [-self.wkspace_width - 0.6, 0.0, self.wkspace_height + 0.3]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(
                self.flex_robot_helper,
                os.path.join(BASE_DIR, "env/deformable_env/src/sim/assets/xarm/xarm6_with_gripper.urdf"),
                robot_base_pos,
                robot_base_orn,
                globalScaling=10.0,
            )
            self.rest_joints = np.zeros(8)
        elif self.obj in ["cloth"]:
            # gripper
            robot_base_pos = [-self.wkspace_width - 0.6, 0.0, self.wkspace_height + 1.0]
            robot_base_orn = [0, 0, 0, 1]
            self.robotId = pyflex.loadURDF(
                self.flex_robot_helper,
                os.path.join(BASE_DIR, "env/deformable_env/src/sim/assets/xarm/xarm6_with_gripper_grasp.urdf"),
                robot_base_pos,
                robot_base_orn,
                globalScaling=10.0,
            )
            self.rest_joints = np.zeros(13)

    def store_data(self, store_cam_param=False, init_fps=False):
        saved_particles = False
        img_list = []
        for j in range(len(self.camPos_list)):
            pyflex.set_camPos(self.camPos_list[j])
            pyflex.set_camAngle(self.camAngle_list[j])

            if store_cam_param:
                self.cam_intrinsic_params[j], self.cam_extrinsic_matrix[j] = (
                    self.camera.get_cam_params()
                )

            # save images
            img = self.render()
            img_list.append(img)

            # save particles
            if not saved_particles:
                # save particle pos
                particles = self.get_positions().reshape(-1, 4)
                particles_pos = particles
                # particles_pos = particles[:, :3] #!!! this is changed
                if self.fps:
                    if init_fps:
                        _, self.sampled_idx = fps_with_idx(
                            particles_pos, self.fps_number
                        )
                    particles_pos = particles_pos[self.sampled_idx]
                self.particle_pos_list.append(particles_pos)
                # save eef pos
                robot_shape_states = pyflex.getRobotShapeStates(self.flex_robot_helper)
                if self.gripper:
                    eef_states = np.zeros((2, 14))
                    eef_states[0] = robot_shape_states[9]  # left finger
                    eef_states[1] = robot_shape_states[12]  # right finger
                else:
                    eef_states = np.zeros((1, 14))
                    eef_states[0] = robot_shape_states[-1]  # pusher
                self.eef_states_list.append(eef_states)

                saved_particles = True

        img_list_np = np.array(img_list)
        self.imgs_list.append(img_list_np)
        self.count += 1

    ### setup gripper
    def _set_pos(self, picker_pos, particle_pos):
        shape_states = np.array(pyflex.get_shape_states()).reshape(-1, 14)
        shape_states[:, 3:6] = shape_states[:, :3]  # picker_pos
        shape_states[:, :3] = picker_pos
        pyflex.set_shape_states(shape_states)
        pyflex.set_positions(particle_pos)

    def _reset_pos(self, particle_pos):
        pyflex.set_positions(particle_pos)

    def robot_close_gripper(self, close, jointPoses=None):
        for j in range(8, self.num_joints):
            pyflex.resetJointState(self.flex_robot_helper, j, close)
        pyflex.set_shape_states(
            self.robot_to_shape_states(
                pyflex.getRobotShapeStates(self.flex_robot_helper)
            )
        )

    def robot_open_gripper(self):
        for j in range(8, self.num_joints):
            pyflex.resetJointState(self.flex_robot_helper, j, 0.0)
    
    def reset_partial(self, save_data=False):
        pyflex.set_shape_states(
            self.robot_to_shape_states(
                pyflex.getRobotShapeStates(self.flex_robot_helper)
            )
        )

        ## update robot shape states
        for idx, joint in enumerate(self.rest_joints):
            pyflex.set_shape_states(
                self.robot_to_shape_states(
                    pyflex.resetJointState(self.flex_robot_helper, idx, joint)
                )
            )

        self.num_joints = p.getNumJoints(self.robotId)
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotId, i)
            jointType = info[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                dof_idx += 1
        self.reset_robot()

        # initial render
        for _ in range(200):
            pyflex.step()

        # save initial rendering
        if save_data:
            self.store_data(store_cam_param=True, init_fps=True)

        # output
        out_data = self.imgs_list, self.particle_pos_list, self.eef_states_list

        return out_data

    ### reset env
    def reset(self, save_data=False):
        pyflex.set_shape_states(
            self.robot_to_shape_states(
                pyflex.getRobotShapeStates(self.flex_robot_helper)
            )
        )

        ## update robot shape states
        for idx, joint in enumerate(self.rest_joints):
            pyflex.set_shape_states(
                self.robot_to_shape_states(
                    pyflex.resetJointState(self.flex_robot_helper, idx, joint)
                )
            )

        self.num_joints = p.getNumJoints(self.robotId)
        self.joints_lower = np.zeros(self.num_dofs)
        self.joints_upper = np.zeros(self.num_dofs)
        dof_idx = 0
        for i in range(self.num_joints):
            info = p.getJointInfo(self.robotId, i)
            jointType = info[2]
            if jointType == p.JOINT_PRISMATIC or jointType == p.JOINT_REVOLUTE:
                self.joints_lower[dof_idx] = info[8]
                self.joints_upper[dof_idx] = info[9]
                dof_idx += 1
        self.reset_robot()

        # initial render
        for _ in range(200):
            pyflex.step()

        # save initial rendering
        if save_data:
            self.store_data(store_cam_param=True, init_fps=True)

        # output
        out_data = self.imgs_list, self.particle_pos_list, self.eef_states_list

        return out_data

    def step(self, action, save_data=False, data=None):
        """
        action: [start_x, start_z, end_x, end_z]
        """
        self.count = 0
        self.imgs_list, self.particle_pos_list, self.eef_states_list = data

        # set up action
        h = 0.5 + self.stick_len
        s_2d = np.concatenate([action[:2], [h]])
        e_2d = np.concatenate([action[2:], [h]])

        # pusher angle depending on x-axis
        if (s_2d - e_2d)[0] == 0:
            pusher_angle = np.pi / 2
        else:
            pusher_angle = np.arctan((s_2d - e_2d)[1] / (s_2d - e_2d)[0])
        # robot orientation
        orn = np.array([0.0, np.pi, pusher_angle + np.pi / 2])

        # create way points
        if self.gripper:
            way_points = [
                s_2d + [0.0, 0.0, 0.5],
                s_2d,
                s_2d,
                e_2d + [0.0, 0.0, 0.5],
                e_2d,
            ]
        else:
            way_points = [s_2d + [0.0, 0.0, 0.2], s_2d, e_2d, e_2d + [0.0, 0.0, 0.2]]
        self.reset_robot(self.rest_joints)

        # set robot speed
        speed = 1.0 / self.robot_speed_inv

        # step
        for i_p in range(len(way_points) - 1):
            s = way_points[i_p]
            e = way_points[i_p + 1]
            steps = int(np.linalg.norm(e - s) / speed) + 1

            for i in range(steps):
                end_effector_pos = s + (e - s) * i / steps  # expected eef position
                end_effector_orn = p.getQuaternionFromEuler(orn)
                jointPoses = p.calculateInverseKinematics(
                    self.robotId,
                    self.end_idx,
                    end_effector_pos,
                    end_effector_orn,
                    self.joints_lower.tolist(),
                    self.joints_upper.tolist(),
                    (self.joints_upper - self.joints_lower).tolist(),
                    self.rest_joints,
                )
                # print('jointPoses:', jointPoses)
                self.reset_robot(jointPoses)
                pyflex.step()

                ## ================================================================
                ## gripper control
                if self.gripper and i_p >= 1:
                    grasp_thresd = 0.1
                    obj_pos = self.get_positions().reshape(-1, 4)[:, :3]
                    new_particle_pos = self.get_positions().reshape(-1, 4).copy()

                    ### grasping
                    if i_p == 1:
                        close = 0
                        start = 0
                        end = 0.7
                        close_steps = 50  # 500
                        finger_y = 0.5
                        for j in range(close_steps):
                            robot_shape_states = pyflex.getRobotShapeStates(
                                self.flex_robot_helper
                            )  # 9: left finger; 12: right finger
                            left_finger_pos, right_finger_pos = (
                                robot_shape_states[9][:3],
                                robot_shape_states[12][:3],
                            )
                            left_finger_pos[1], right_finger_pos[1] = (
                                left_finger_pos[1] - finger_y,
                                right_finger_pos[1] - finger_y,
                            )
                            new_finger_pos = (left_finger_pos + right_finger_pos) / 2

                            if j == 0:
                                # fine the k pick point
                                pick_k = 5
                                left_min_dist, left_pick_index = find_min_distance(
                                    left_finger_pos, obj_pos, pick_k
                                )
                                right_min_dist, right_pick_index = find_min_distance(
                                    right_finger_pos, obj_pos, pick_k
                                )
                                min_dist, pick_index = find_min_distance(
                                    new_finger_pos, obj_pos, pick_k
                                )
                                # save the original setting for restoring
                                pick_origin = new_particle_pos[pick_index]

                            if (
                                left_min_dist <= grasp_thresd
                                or right_min_dist <= grasp_thresd
                            ):
                                new_particle_pos[left_pick_index, :3] = left_finger_pos
                                new_particle_pos[left_pick_index, 3] = 0
                                new_particle_pos[right_pick_index, :3] = (
                                    right_finger_pos
                                )
                                new_particle_pos[right_pick_index, 3] = 0
                            self._set_pos(new_finger_pos, new_particle_pos)

                            # close the gripper
                            close += (end - start) / close_steps
                            self.robot_close_gripper(close)
                            pyflex.step()

                    # find finger positions
                    robot_shape_states = pyflex.getRobotShapeStates(
                        self.flex_robot_helper
                    )  # 9: left finger; 12: right finger
                    left_finger_pos, right_finger_pos = (
                        robot_shape_states[9][:3],
                        robot_shape_states[12][:3],
                    )
                    left_finger_pos[1], right_finger_pos[1] = (
                        left_finger_pos[1] - finger_y,
                        right_finger_pos[1] - finger_y,
                    )
                    new_finger_pos = (left_finger_pos + right_finger_pos) / 2
                    # connect pick pick point to the finger
                    new_particle_pos[pick_index, :3] = new_finger_pos
                    new_particle_pos[pick_index, 3] = 0
                    self._set_pos(new_finger_pos, new_particle_pos)

                    self.reset_robot(jointPoses)
                    pyflex.step()

                ## ================================================================

                # save img in each step
                obj_pos = self.get_positions().reshape(-1, 4)[:, [0, 2]]
                obj_pos[:, 1] *= -1
                robot_obj_dist = np.min(
                    cdist(end_effector_pos[:2].reshape(1, 2), obj_pos)
                )
                if save_data:
                    rob_obj_dist_thresh = self.dataset_config["rob_obj_dist_thresh"]
                    contact_interval = self.dataset_config["contact_interval"]
                    non_contact_interval = self.dataset_config["non_contact_interval"]
                    if (
                        robot_obj_dist < rob_obj_dist_thresh
                        and i % contact_interval == 0
                    ):  # robot-object contact
                        self.store_data()
                    elif i % non_contact_interval == 0:  # not contact
                        self.store_data()

                self.reset_robot()
                if math.isnan(self.get_positions().reshape(-1, 4)[:, 0].max()):
                    print("simulator exploded when action is", action)
                    return None

        # set up gripper
        if self.gripper:
            self.robot_open_gripper()
            # reset the mass for the pick points
            new_particle_pos[pick_index, 3] = pick_origin[:, 3]
            self._reset_pos(new_particle_pos)

        self.reset_robot()

        for i in range(200):
            pyflex.step()

        # save final rendering
        if save_data:
            self.store_data()

        obs = self.render()
        out_data = self.imgs_list, self.particle_pos_list, self.eef_states_list

        return obs, out_data

    def render(self, no_return=False):
        pyflex.step()
        if no_return:
            return
        else:
            return pyflex.render(render_depth=True).reshape(self.screenHeight, self.screenWidth, 5)

    def close(self):
        pyflex.clean()

    def sample_action(self, init=False, boundary_points=None, boundary=None):
        if self.obj in ["rope", "granular"]:
            action = self.sample_deform_actions()
            return action
        elif self.obj in ["cloth"]:
            action, boundary_points, boundary = self.sample_grasp_actions_corner(
                init, boundary_points, boundary
            )
            return action, boundary_points, boundary
        else:
            raise ValueError("action not defined")

    def sample_deform_actions(self):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1  # align with the coordinates
        num_points = positions.shape[0]
        pos_xz = positions[:, [0, 2]]

        pos_x, pos_z = positions[:, 0], positions[:, 2]
        center_x, center_z = np.median(pos_x), np.median(pos_z)
        chosen_points = []
        for idx, (x, z) in enumerate(zip(pos_x, pos_z)):
            if np.sqrt((x - center_x) ** 2 + (z - center_z) ** 2) < 2.0:
                chosen_points.append(idx)
        # print(f'chosen points {len(chosen_points)} out of {num_points}.')
        if len(chosen_points) == 0:
            print("no chosen points")
            chosen_points = np.arange(num_points)

        # random choose a start point which can not be overlapped with the object
        valid = False
        for _ in range(1000):
            startpoint_pos_origin = np.random.uniform(
                -self.action_space, self.action_space, size=(1, 2)
            )
            startpoint_pos = startpoint_pos_origin.copy()
            startpoint_pos = startpoint_pos.reshape(-1)

            # choose end points which is the expolation of the start point and obj point
            pickpoint = np.random.choice(chosen_points)
            obj_pos = positions[pickpoint, [0, 2]]
            slope = (obj_pos[1] - startpoint_pos[1]) / (obj_pos[0] - startpoint_pos[0])
            if obj_pos[0] < startpoint_pos[0]:
                # 1.0 for planning
                # (1.5, 2.0) for data collection
                x_end = obj_pos[0] - 1.0  # rand_float(1.5, 2.0)
            else:
                x_end = obj_pos[0] + 1.0  # rand_float(1.5, 2.0)
            y_end = slope * (x_end - startpoint_pos[0]) + startpoint_pos[1]
            endpoint_pos = np.array([x_end, y_end])
            if (
                obj_pos[0] != startpoint_pos[0]
                and np.abs(x_end) < 1.5
                and np.abs(y_end) < 1.5
                and np.min(cdist(startpoint_pos_origin, pos_xz)) > 0.2
            ):
                valid = True
                break

        if valid:
            action = np.concatenate(
                [startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0
            )
        else:
            action = None

        return action

    def sample_grasp_actions_corner(
        self, init=False, boundary_points=None, boundary=None
    ):
        positions = self.get_positions().reshape(-1, 4)
        positions[:, 2] *= -1
        particle_x, particle_y, particle_z = (
            positions[:, 0],
            positions[:, 1],
            positions[:, 2],
        )
        x_min, y_min, z_min = np.min(particle_x), np.min(particle_y), np.min(particle_z)
        x_max, y_max, z_max = np.max(particle_x), np.max(particle_y), np.max(particle_z)

        # choose the starting point at the boundary of the object
        if init:  # record boundary points
            boundary_points = []
            boundary = []
            for idx, point in enumerate(positions):
                if point[0] == x_max:
                    boundary_points.append(idx)
                    boundary.append(1)
                elif point[0] == x_min:
                    boundary_points.append(idx)
                    boundary.append(2)
                elif point[2] == z_max:
                    boundary_points.append(idx)
                    boundary.append(3)
                elif point[2] == z_min:
                    boundary_points.append(idx)
                    boundary.append(4)
        assert len(boundary_points) == len(boundary)

        # random pick a point as start point
        valid = False
        for _ in range(1000):
            pick_idx = np.random.choice(len(boundary_points))
            startpoint_pos = positions[boundary_points[pick_idx], [0, 2]]
            endpoint_pos = startpoint_pos.copy()
            # choose end points which is outside the obj
            move_distance = rand_float(1.0, 1.5)

            if boundary[pick_idx] == 1:
                endpoint_pos[0] += move_distance
            elif boundary[pick_idx] == 2:
                endpoint_pos[0] -= move_distance
            elif boundary[pick_idx] == 3:
                endpoint_pos[1] += move_distance
            elif boundary[pick_idx] == 4:
                endpoint_pos[1] -= move_distance

            if np.abs(endpoint_pos[0]) < 3.5 and np.abs(endpoint_pos[1]) < 2.5:
                valid = True
                break

        if valid:
            action = np.concatenate(
                [startpoint_pos.reshape(-1), endpoint_pos.reshape(-1)], axis=0
            )
        else:
            action = None

        return action, boundary_points, boundary

    def get_positions(self):
        return pyflex.get_positions()

    def set_positions(self, positions):
        pyflex.set_positions(positions)

    def get_num_particles(self):
        return self.get_positions().reshape(-1, 4).shape[0]

    def get_property_params(self):
        return self.scene.get_property_params()

    def get_states(self):
        return self.get_positions().reshape(-1, 4)

    def set_states(self, states):
        if states is not None:
            # self.scene.set_scene(self.obj)
            pyflex.set_positions(states)

    def seed(self, seed):
        np.random.seed(seed)

    def get_one_view_img(self, cam_id=None):
        cam_id = cam_id or self.camera_view
        pyflex.set_camPos(self.camPos_list[cam_id])
        pyflex.set_camAngle(self.camAngle_list[cam_id])
        return self.render()
