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
import random
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

    imgname_arr = cliff_gt["imgname"]
    imgname_arr_unique = np.unique(imgname_arr)
    imgname_arr_unique = sorted(imgname_arr_unique)
    # imgname_arr_unique = sorted(imgname_arr_unique, reverse=True)
    # random.shuffle(imgname_arr_unique)

    global target_index
    target_index = 0
    while True:
        # iterate each image
        target_imgname = imgname_arr_unique[target_index]
        # specify image name
        # target_imgname = "train2014/COCO_train2014_000000003077.jpg"
        # target_imgname = "images/000845150.jpg"
        # target_imgname = "images/000460055.jpg"
        print("image name:", target_imgname)

        # load the image
        img_path = osp(img_dir, target_imgname)
        img = cv2.imread(img_path)
        img_h, img_w, _ = np.shape(img)

        smpl_vertices_list = []
        target_focal_l = -1
        skeleton2d_img = img.copy()
        for imgname, pose, shape, global_t, focal_l, part in \
                zip(cliff_gt["imgname"], cliff_gt["pose"], cliff_gt["shape"],
                    cliff_gt["global_t"], cliff_gt["focal_l"], cliff_gt["part"]):
            if imgname != target_imgname:
                continue

            target_focal_l = focal_l
            # get mesh vertices using the SMPL model
            pose = torch.FloatTensor(pose).view(1, -1)      # 1*72
            shape = torch.FloatTensor(shape).view(1, -1)    # 1*10
            global_t = torch.FloatTensor(global_t).view(1, -1)    # 1*3
            smpl_output = smpl_model(betas=shape, pose2rot=True,
                                     body_pose=pose[:, 3:],
                                     global_orient=pose[:, :3],
                                     transl=global_t)
            smpl_vertices = smpl_output.vertices[0].numpy()
            smpl_vertices_list.append(smpl_vertices)

            # ------------ draw the 2D keypoint skeleton ------------
            skeleton2d_img = draw_skeleton(skeleton2d_img, part)

        # setup the render
        renderer = Renderer(focal_length=target_focal_l, img_w=img_w, img_h=img_h, faces=smpl_model.faces)

        smpl_vertices_arr = np.array(smpl_vertices_list)
        # ------------ render the front view ------------
        front_view_img = renderer.render_front_view(smpl_vertices_arr, bg_img_rgb=img[:, :, ::-1].copy())

        # ------------ render the side view ------------
        side_view_img = renderer.render_side_view(smpl_vertices_arr)

        # delete the render
        renderer.delete()

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
        plt.suptitle(target_imgname)
        plt.tight_layout()
        fig.canvas.mpl_connect("key_press_event", on_key_press)
        plt.show()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--img_dir', help='the image directory')
    parser.add_argument('--cliffGT_path', help='path to the pseudo-GT file')
    args = parser.parse_args()

    main(args.img_dir, args.cliffGT_path, SMPL_MODEL_DIR)
