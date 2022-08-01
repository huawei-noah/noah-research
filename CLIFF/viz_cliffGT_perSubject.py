# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.

# This program is free software; you can redistribute it and/or modify it
# under the terms of the MIT license.

# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.

import cv2
import smplx
import torch
import argparse
import numpy as np
from os.path import join as osp
import matplotlib.pyplot as plt

from common.renderer_pyrd import Renderer
from common.skeleton_drawer import draw_skeleton
from common.constants import SMPL_MODEL_DIR


def main(img_dir, cliffGT_path, smpl_path):
    # setup the SMPL model
    smpl_model = smplx.create(smpl_path, "smpl")

    cliff_gt = np.load(cliffGT_path)
    print("cliff_gt keys:", list(cliff_gt.keys()))

    global target_index
    target_index = 0
    while True:
        # iterate each subject
        imgname = cliff_gt["imgname"][target_index]
        pose = cliff_gt["pose"][target_index]
        shape = cliff_gt["shape"][target_index]
        global_t = cliff_gt["global_t"][target_index]
        focal_l = cliff_gt["focal_l"][target_index]
        part = cliff_gt["part"][target_index]
        print("image name:", imgname)

        # get mesh vertices using the SMPL model
        pose = torch.FloatTensor(pose).view(1, -1)      # 1*72
        shape = torch.FloatTensor(shape).view(1, -1)    # 1*10
        global_t = torch.FloatTensor(global_t).view(1, -1)    # 1*3
        smpl_output = smpl_model(betas=shape, pose2rot=True,
                                 body_pose=pose[:, 3:],
                                 global_orient=pose[:, :3],
                                 transl=global_t)
        smpl_vertices = smpl_output.vertices.numpy()

        # load the image
        img_path = osp(img_dir, imgname)
        img = cv2.imread(img_path)
        img_h, img_w, _ = np.shape(img)

        # setup the render
        renderer = Renderer(focal_length=focal_l, img_w=img_w, img_h=img_h, faces=smpl_model.faces)
        # ------------ render the front view ------------
        front_view_img = renderer.render_front_view(smpl_vertices, bg_img_rgb=img[:, :, ::-1].copy())
        # ------------ render the side view ------------
        side_view_img = renderer.render_side_view(smpl_vertices)
        # delete the render
        renderer.delete()

        # ------------ draw the 2D keypoint skeleton ------------
        skeleton2d_img = draw_skeleton(img.copy(), part)

        # ------------ show the images ------------
        def on_key_press(event):
            global target_index
            if event.key == "right":
                target_index += 1
            elif event.key == "left":
                target_index -= 1
            elif event.key == "escape":
                exit()
            plt.close()

        images = [img[:, :, ::-1], skeleton2d_img[:, :, ::-1],
                  front_view_img, side_view_img]
        titles = ["image", "2D skeleton", "front view", "side view"]
        fig = plt.figure()
        for idx, (image, title) in enumerate(zip(images, titles)):
            fig.add_subplot(2, 2, idx + 1)
            plt.imshow(image)
            plt.title(title)
            plt.axis("off")
        plt.suptitle(imgname)
        plt.tight_layout()
        fig.canvas.mpl_connect("key_press_event", on_key_press)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='the image directory')
    parser.add_argument('--cliffGT_path', help='path to the pseudo-GT file')
    args = parser.parse_args()

    main(args.img_dir, args.cliffGT_path, SMPL_MODEL_DIR)
