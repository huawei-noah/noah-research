# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import numpy as np
import cv2

"""
Joint Order

0:  'Right Ankle'
1:  'Right Knee'
2:  'Right Hip'
3:  'Left Hip'
4:  'Left Knee'
5:  'Left Ankle'
6:  'Right Wrist'
7:  'Right Elbow'
8:  'Right Shoulder'
9:  'Left Shoulder'
10: 'Left Elbow'
11: 'Left Wrist'
12: 'Neck (LSP)'
13: 'Top of Head (LSP)'
14: 'Pelvis (MPII)'
15: 'Thorax (MPII)'
16: 'Spine (H36M)'
17: 'Jaw (H36M)'
18: 'Head (H36M)'
19: 'Nose'
20: 'Left Eye'
21: 'Right Eye'
22: 'Left Ear'
23: 'Right Ear'
"""

palette = np.array([[255, 128, 0], [255, 153, 51], [255, 178, 102],
                    [230, 230, 0], [255, 153, 255], [153, 204, 255],
                    [255, 102, 255], [255, 51, 255], [102, 178, 255],
                    [51, 153, 255], [255, 153, 153], [255, 102, 102],
                    [255, 51, 51], [153, 255, 153], [102, 255, 102],
                    [51, 255, 51], [0, 255, 0], [0, 0, 255], [255, 0, 0],
                    [255, 255, 255]])  # BGR

# --------------- COCO ---------------
skeleton_coco = [[9, 8], [8, 2], [2, 3], [3, 9],  # torso
                 [9, 10], [10, 11],  # left arm
                 [8, 7], [7, 6],  # right arm
                 [2, 1], [1, 0],  # right leg
                 [3, 4], [4, 5],  # left leg
                 [23, 21], [21, 19], [19, 20], [20, 22], [20, 21],  # face
                 [9, 22], [8, 23], ]  # ear and shoulder
skeleton_coco_color = palette[[9, 9, 9, 9,  # torso, orange
                               7, 7, 0, 0, 0, 0, 7, 7,  # limb, blue(0), pink(7)
                               16, 16, 16, 16, 16,  # face, green
                               16, 16]]  # ear and shoulder, green
joint_coco_color = palette[[0, 0, 0, 7, 7, 7, 0, 0, 0, 7, 7, 7,
                            9, 9, 9, 9, 9, 9, 9,
                            16, 16, 16, 16, 16]]

# --------------- MPII ---------------
skeleton_mpii = [[14, 15], [15, 12], [12, 13],  # torso
                 [15, 9], [9, 10], [10, 11],  # left arm
                 [15, 8], [8, 7], [7, 6],  # right arm
                 [14, 2], [2, 1], [1, 0],  # right leg
                 [14, 3], [3, 4], [4, 5]]  # left leg
skeleton_mpii_color = palette[[9, 9, 9,  # torso, orange
                               7, 7, 7,
                               0, 0, 0,
                               0, 0, 0,
                               7, 7, 7]]  # limb, blue(0), pink(7)
joint_mpii_color = palette[[0, 0, 0, 7, 7, 7, 0, 0, 0, 7, 7, 7,
                            9, 9, 9, 9, 9, 9, 9,
                            16, 16, 16, 16, 16]]


def draw_skeleton(img, kp_24joints):
    # -------- check the format --------
    # COCO format
    skeleton = skeleton_coco
    skeleton_color = skeleton_coco_color
    joint_color = joint_coco_color
    if kp_24joints[13, 2] > 0:
        # MPII format
        skeleton = skeleton_mpii
        skeleton_color = skeleton_mpii_color
        joint_color = joint_mpii_color

    # -------- draw lines --------
    for k in np.arange(len(skeleton)):
        parent = skeleton[k][0]
        child = skeleton[k][1]
        # there are missing joint in this bone
        if kp_24joints[parent][2] * kp_24joints[child][2] <= 0:
            continue
        parent_pos = (int(kp_24joints[parent][0]), int(kp_24joints[parent][1]))
        child_pos = (int(kp_24joints[child][0]), int(kp_24joints[child][1]))
        color = tuple([int(x) for x in skeleton_color[k]])
        cv2.line(img, parent_pos, child_pos, color, 3)

    # -------- draw joints --------
    for index, (px, py, conf) in enumerate(kp_24joints):
        if conf > 0:
            color = tuple([int(x) for x in joint_color[index]])
            cv2.circle(img, (int(px), int(py)), 5, color, -1)

    return img
