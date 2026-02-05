from typing import NamedTuple, Optional
from dataclasses import dataclass
import random
import math

import torch

from .single import DotDataset, DotDatasetConfig
from .wall_utils import *


class WallSample(NamedTuple):
    states: torch.Tensor  # [(batch_size), T, 1, 28, 28]
    locations: torch.Tensor  # [(batch_size), T, 2]
    actions: torch.Tensor  # [(batch_size), T, 2]
    bias_angle: torch.Tensor  # [(batch_size), 2]
    wall_x: torch.Tensor  # [(batch_size), 1]
    door_y: torch.Tensor  # [(batch_size), 1]


@dataclass
class WallDatasetConfig(DotDatasetConfig):
    fix_wall: bool = True
    fix_wall_batch_k: Optional[int] = None

    fix_wall: bool = True
    wall_padding: int = 20
    door_padding: int = 10
    wall_width: int = 1
    door_space: int = 2
    exclude_wall_train: str = ""  # don't generate wall at these x-axis values
    exclude_door_train: str = ""  # don't generate door at these y-axis values
    only_wall_val: str = ""  # only evalaute wall at these x-axis values
    only_door_val: str = ""  # only evaluate door at these y-axis values
    fix_wall_location: Optional[int] = None
    fix_door_location: Optional[int] = None
    num_train_layouts: Optional[int] = -1


class WallDataset(DotDataset):
    def __init__(
        self,
        config: WallDatasetConfig,
        *args,
        **kwargs,
    ):
        layouts, other_layouts = generate_wall_layouts(config)
        self.layouts = layouts
        super().__init__(config, *args, **kwargs)

    def render_location(self, locations):
        states = super().render_location(locations)
        return states

    def generate_cross_wall_points(self, wall_locs, action_padding=0):
        """
        Parameters:
            wall_locs (bs)
        Output:
            starts (bs, 2)
            goal (bs, 2)
        Description:
            Generate left and right points on opposite sides of the wall
        """
        bs = wall_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2

        min_val = self.config.border_wall_loc - 1 + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01

        left_x = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            left_wall_locs - action_padding,
        )

        right_x = sample_uniformly_between(
            right_wall_locs + action_padding,
            torch.full((bs,), max_val).to(wall_locs.device),
        )

        left_y = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            torch.full((bs,), max_val).to(wall_locs.device),
        )

        right_y = sample_uniformly_between(
            torch.full((bs,), min_val).to(wall_locs.device),
            torch.full((bs,), max_val).to(wall_locs.device),
        )

        left_pos = torch.stack([left_x, left_y]).transpose(0, 1)
        right_pos = torch.stack([right_x, right_y]).transpose(0, 1)

        return left_pos, right_pos

    def generate_cross_wall_state_and_actions(
        self,
        wall_locs=None,
        door_locs=None,
        n_steps=17,
    ):
        """
        Parameters:
            wall_locs (bs)
            door_locs (bs)
            actions (bs)
            n_steps: int
        Output:
            location (bs, 2)
            actions (bs, n_steps-1, 2)
            bias_angle (bs, 2)
        """
        bs = door_locs.size(0)
        left_wall_locs = wall_locs - self.config.wall_width // 2
        right_wall_locs = wall_locs + self.config.wall_width // 2

        # sample point at door
        x = sample_uniformly_between(
            left_wall_locs,
            right_wall_locs,
        )

        # we define a truncated normal distribution to sample y centered at the door center
        y = sample_truncated_norm(
            upper_bound=door_locs + self.config.door_space,
            lower_bound=door_locs - self.config.door_space,
            mean=door_locs,
        ).to(door_locs.device)

        loc_at_door = torch.stack([x, y]).transpose(0, 1)
        # sample the step which this point refers to
        step_idxs = torch.randint(1, n_steps, size=x.shape)

        # determine the angles for points above
        angles = torch.empty(bs)

        # Sample angles pointing left
        for i in range(bs):
            angles[i] = torch.pi + (torch.rand(1) - 0.5) * torch.pi / 2

        angles = self.angle_to_vec(angles).to(self.device)
        actions_dir_left, _ = self.generate_actions(n_steps, bias_angle=angles)
        actions_dir_right, _ = self.generate_actions(n_steps, bias_angle=-1 * angles)

        cw_actions = torch.zeros((bs, n_steps - 1, 2))
        cw_start_loc = torch.zeros((bs, 2))

        for i in range(bs):
            step = step_idxs[i]

            # right pointing trajectory
            traj = torch.cat(
                [
                    torch.flip(actions_dir_left[i][:step], dims=[0]) * -1,
                    actions_dir_right[i][1 : n_steps - step],
                ]
            )
            step_sum_before_door = traj[:step].sum(dim=0)

            if random.random() < 0.5:
                # turn it into left pointing trajectory
                traj = torch.flip(traj, dims=[0]) * -1
                step_sum_before_door = traj[: n_steps - step].sum(dim=0)

            cw_actions[i] = traj

            # calcualte start position such that action at given step will reach loc at door
            cw_start_loc[i] = loc_at_door[i] - step_sum_before_door

        min_val = self.config.border_wall_loc - 1 + 0.01
        max_val = self.config.img_size - self.config.border_wall_loc - 0.01
        cw_start_loc = torch.clamp(cw_start_loc, min=min_val, max=max_val)

        return cw_start_loc, cw_actions, torch.zeros_like(cw_actions)

    def check_wall_intersection(self, current_location, next_location, walls):
        """
        Args:
            current_location (bs, 2):
            next_location (bs, 2):
            walls (bs): x coordinate of walls
        """

        # Calculate the half width to determine the range of the wall
        half_width = self.config.wall_width // 2

        # Calculate the left and right boundaries of the walls
        wall_left = walls - half_width
        wall_right = walls + half_width

        # Determine the relative positions of the current and next locations to the wall's boundaries
        # Check if the x-coordinates are less than the right boundary of the wall
        current_right = current_location[:, 0] <= wall_right
        next_right = next_location[:, 0] <= wall_right

        # Check if the x-coordinates are more than the left boundary of the wall
        current_left = current_location[:, 0] >= wall_left
        next_left = next_location[:, 0] >= wall_left

        # Evaluate intersection conditions:
        # Case 1: One point is inside the wall boundaries and the other is outside
        # Case 2: Both points are outside the wall boundaries but on opposite sides of the wall
        inside_wall = (current_right & current_left) != (next_right & next_left)
        across_wall = (current_right != next_right) & (current_left != next_left)

        check_wall_intersection = inside_wall | across_wall

        return check_wall_intersection

    def check_pass_through_door(
        self, current_location, next_location, wall_loc, door_loc
    ):
        """
        Args:
            current_location (2,):
            next_location (2,):
            wall_loc (1,):
            door_loc (1,):
        Summary:
            By this point we assume the path intersects a wall
            This function finds out whether if the intersection happens at the door
        """
        half_width = self.config.wall_width // 2

        # Calculate intersection points with the left and right boundaries of the wall
        left_wall = wall_loc - half_width
        right_wall = wall_loc + half_width

        # Get the displacement vector
        d = next_location - current_location

        # Calculate the slope (a) and intercept (b) of the line
        a = d[1] / d[0]
        b = current_location[1] - a * current_location[0]

        # if path intersects left wall
        if (
            torch.sign(left_wall - current_location[0])
            * torch.sign(left_wall - next_location[0])
            < 0
        ):
            # calcualte y coordinate of intersection point with left wall
            y_left = a * left_wall + b
            pass_left_wall = (
                door_loc - self.config.door_space
                <= y_left
                <= door_loc + self.config.door_space
            )
        else:
            pass_left_wall = True

        # if path intersects right wall
        if (
            torch.sign(right_wall - current_location[0])
            * torch.sign(right_wall - next_location[0])
            < 0
        ):
            # calculate y coordinate of intersection point with right wall
            y_right = a * right_wall + b
            pass_right_wall = (
                door_loc - self.config.door_space
                <= y_right
                <= door_loc + self.config.door_space
            )
        else:
            pass_right_wall = True

        return pass_left_wall and pass_right_wall

    @staticmethod
    def segments_intersect(A, B):
        """
        Input:
            A: (bs, 2, 2)
            B: (bs, 2, 2)
        Summary:
            Test whether if the segment from A[i][0] to A[i][1] intersects
            the segment from B[i][0] to B[i][1]
        """
        # Extract points
        A0, A1 = A[:, 0], A[:, 1]  # Endpoints of segment A
        B0, B1 = B[:, 0], B[:, 1]  # Endpoints of segment B

        # Direction vectors
        dA = A1 - A0  # Direction vector of segment A
        dB = B1 - B0  # Direction vector of segment B

        # Helper function to compute cross product of 2D vectors
        def cross_2d(v, w):
            return v[:, 0] * w[:, 1] - v[:, 1] * w[:, 0]

        # Translate points to origin based on one endpoint of each segment
        # Check orientation of other segment's endpoints relative to this segment
        B0_to_A0 = B0 - A0
        B1_to_A0 = B1 - A0
        A0_to_B0 = A0 - B0
        A1_to_B0 = A1 - B0

        # Cross products to determine the relative positions
        cross_A_B0 = cross_2d(dA, B0_to_A0)
        cross_A_B1 = cross_2d(dA, B1_to_A0)
        cross_B_A0 = cross_2d(dB, A0_to_B0)
        cross_B_A1 = cross_2d(dB, A1_to_B0)

        # Intersection condition: opposite signs of cross products indicate the points are on opposite sides
        intersect_A = cross_A_B0 * cross_A_B1 < 0  # B endpoints on opposite sides of A
        intersect_B = cross_B_A0 * cross_B_A1 < 0  # A endpoints on opposite sides of B

        # Combine conditions for full intersection test
        # Use logical AND: both conditions must be true for segments to intersect
        intersection = intersect_A & intersect_B

        # Return results as 1 (intersect) or 0 (no intersect)
        return intersection.long()

    def check_wall_width_intersection(
        self,
        locations,
        next_locations,
        walls,
        doors,
    ):
        disp = torch.stack([locations, next_locations], dim=1)

        # check if the action is pointing upwards or downwards
        deltas = next_locations - locations
        upwards = deltas[:, 1] > 0
        downwards = deltas[:, 1] < 0

        left_wall = walls - self.config.wall_width // 2
        right_wall = walls + self.config.wall_width // 2

        door_bot = doors - self.config.door_space
        door_top = doors + self.config.door_space

        top_left = torch.stack([left_wall, door_top], dim=1)
        top_right = torch.stack([right_wall, door_top], dim=1)

        bot_left = torch.stack([left_wall, door_bot], dim=1)
        bot_right = torch.stack([right_wall, door_bot], dim=1)

        top_seg = torch.stack([top_left, top_right], dim=1)
        bot_seg = torch.stack([bot_left, bot_right], dim=1)

        top_intersect = self.segments_intersect(disp, top_seg)
        bot_intersect = self.segments_intersect(disp, bot_seg)

        output = (top_intersect & upwards) | (bot_intersect & downwards)

        return output

    def generate_transitions(
        self,
        location,
        actions,
        bias_angle,
        walls,
    ):
        """
        Parameters:
            location: [bs, 2]
            actions: [bs, n_steps-1, 2]
            bias_angle: [bs, 2]
            walls: tuple([bs], [bs])
        """
        locations = [location]
        for i in range(actions.shape[1]):
            next_location = self.generate_transition(locations[-1], actions[:, i])
            left_border = torch.zeros_like(walls[0])
            left_border[:] = self.config.border_wall_loc - 1
            right_border = torch.zeros_like(walls[0])
            right_border[:] = self.config.img_size - self.config.border_wall_loc
            top_border, bot_border = left_border, right_border

            check_border_intersection = (
                (
                    (
                        torch.sign(locations[-1][:, 0] - left_border)
                        * torch.sign(next_location[:, 0] - left_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 0] - right_border)
                        * torch.sign(next_location[:, 0] - right_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 1] - top_border)
                        * torch.sign(next_location[:, 1] - top_border)
                    )
                    <= 0
                )
                | (
                    (
                        torch.sign(locations[-1][:, 1] - bot_border)
                        * torch.sign(next_location[:, 1] - bot_border)
                    )
                    <= 0
                )
            )

            check_wall_intersection = self.check_wall_intersection(
                locations[-1], next_location, walls[0]
            )

            check_wall_width_intersection = self.check_wall_width_intersection(
                locations=locations[-1],
                next_locations=next_location,
                walls=walls[0],
                doors=walls[1],
            )

            check_intersection = (
                check_border_intersection
                | check_wall_intersection
                | check_wall_width_intersection
            )

            for j in check_intersection.nonzero():
                if check_border_intersection[j] or check_wall_width_intersection[j]:
                    next_location[j] = locations[-1][j].clone()
                else:
                    if not self.check_pass_through_door(
                        current_location=locations[-1][j][0],
                        next_location=next_location[j][0],
                        wall_loc=walls[0][j],
                        door_loc=walls[1][j],
                    ):
                        next_location[j] = locations[-1][j].clone()

            locations.append(next_location)

        # Unsqueeze for compatibility with multi-dot dataset
        locations = torch.stack(locations, dim=1).unsqueeze(dim=-2)
        actions = actions.unsqueeze(dim=-2)
        states = self.render_location(locations)
        walls = self.render_walls(*walls).unsqueeze(1).unsqueeze(1)
        walls = walls.repeat(1, states.shape[1], 1, 1, 1)
        states_with_walls = torch.cat([states, walls], dim=-3)

        if self.config.n_steps_reduce_factor > 1:
            states_with_walls = states_with_walls[
                :, :: self.config.n_steps_reduce_factor
            ]
            locations = locations[:, :: self.config.n_steps_reduce_factor]
            reduced_chunks = actions.shape[1] // self.config.n_steps_reduce_factor
            action_chunks = torch.chunk(actions, chunks=reduced_chunks, dim=1)
            actions = torch.cat(
                [torch.sum(chunk, dim=1, keepdim=True) for chunk in action_chunks],
                dim=1,
            )

        return WallSample(
            states=states_with_walls,
            locations=locations,
            actions=actions,
            bias_angle=bias_angle,
            wall_x=None,
            door_y=None,
        )

    def sample_walls(self):
        """
        Returns:
        wall_x: Tensor (bs). x coordinate of the wall
        door_y: Tensor (bs). y coordinate of the door
        """
        layout_codes = list(self.layouts.keys())
        if self.config.fix_wall_batch_k is not None:
            layout_codes = random.sample(layout_codes, self.config.fix_wall_batch_k)

        weights = [1] * len(layout_codes)
        sampled_codes = random.choices(
            layout_codes, weights=weights, k=self.config.batch_size
        )
        wall_locs = []
        door_locs = []
        types = []

        for code in sampled_codes:
            attr = self.layouts[code]
            wall_locs.append(attr["wall_pos"])
            door_locs.append(attr["door_pos"])
            types.append(attr["type"])

        wall_locs = torch.tensor(wall_locs, device=self.device)
        door_locs = torch.tensor(door_locs, device=self.device)
        return (wall_locs, door_locs)

    def render_walls(self, wall_locs, hole_locs):
        """
        Params:
            wall_locs: torch tensor size (batch_size,)
                holds x coordinates of walls for each batch index
            hole_locs: torch tensor size (batch_size,)
                holds y coordinates of doors for each batch index
        """
        x = torch.arange(0, self.config.img_size, device=self.device)
        y = torch.arange(0, self.config.img_size, device=self.device)
        grid_x, grid_y = torch.meshgrid(x, y, indexing="xy")
        grid_x = grid_x.unsqueeze(0).repeat(self.config.batch_size, 1, 1)
        grid_y = grid_y.unsqueeze(0).repeat(self.config.batch_size, 1, 1)

        wall_locs_r = wall_locs.view(self.config.batch_size, 1, 1).repeat(
            1, self.config.img_size, self.config.img_size
        )
        hole_locs_r = hole_locs.view(self.config.batch_size, 1, 1).repeat(
            1, self.config.img_size, self.config.img_size
        )

        # Calculate offsets for wall width
        offset = self.config.wall_width // 2
        wall_mask = (wall_locs_r - offset <= grid_x) & (grid_x <= wall_locs_r + offset)

        res = (
            wall_mask
            * (
                (hole_locs_r < grid_y - self.config.door_space)
                + (hole_locs_r > grid_y + self.config.door_space)
            )
        ).float()

        # set border walls
        border_wall_loc = self.config.border_wall_loc
        res[:, :, border_wall_loc - 1] = 1
        res[:, :, -border_wall_loc] = 1
        res[:, border_wall_loc - 1, :] = 1
        res[:, -border_wall_loc, :] = 1

        return res
