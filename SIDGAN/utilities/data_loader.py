#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.
#This program is free software; you can redistribute it and/or modify it under the terms of the BSD 0-Clause License.
#This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
# See the BSD 0-Clause License for more details.

import os
import numpy as np
from PIL import Image
from keras.utils import Sequence
import cv2 as cv2
import random


def normalize_negative_one(img):
    normalized_input = (img - np.amin(img)) / (np.amax(img) - np.amin(img))
    return 2 * normalized_input - 1


def random_crop_b2c_sup(raw_img, long_img, CROP_SIZE):
    i = random.randint(0, raw_img.shape[0] - CROP_SIZE + 1)
    j = random.randint(0, raw_img.shape[1] - CROP_SIZE + 1)
    i = i - (i % 2)
    j = j - (j % 2)
    raw_crop = raw_img[i:(i + CROP_SIZE), j:(j + CROP_SIZE), :]
    long_crop = long_img[i:(i + CROP_SIZE), j:(j + CROP_SIZE), :]
    return raw_crop, long_crop


def random_crop(img, random_crop_size):
    height, width = img.shape[0], img.shape[1]
    dy, dx = random_crop_size
    x = np.random.randint(0, width - dx + 1)
    y = np.random.randint(0, height - dy + 1)
    return img[y:(y + dy), x:(x + dx), :]


def load_data(task, root, batch_size=1, crop_size=256, generator=False):
    if task == 'B2C' or task == 'B2C_sup':
        trainA_image_names, trainB_image_names = load_B2C(root)
    elif task == 'A2B':
        trainA_image_names, trainB_image_names = load_A2B(root)
    if generator:
        return data_sequence(root, trainA_image_names, trainB_image_names, crop_size=crop_size,
                             task=task, batch_size=batch_size)


def load_B2C(root):
    filename = root + '/train_valid_list.txt'
    videos = [line.rstrip('\n') for line in open(filename)]
    trainA_image_names = []
    trainB_image_names = []
    for i in videos:
        imgs = os.listdir(os.path.join(root, 'SID_long', i))
        for x in imgs:
            if x.startswith('0') and x.endswith('.png'):
                trainA_image_names.append(os.path.join(root, 'SID_long', i, x))
        trainB_image_names = []
        for i in videos:
            imgs = os.listdir(os.path.join(root, 'VBM4D_rawRGB', i))
            for x in imgs:
                if x.startswith('0') and x.endswith('.png'):
                    trainB_image_names.append(os.path.join(root, 'VBM4D_rawRGB', i, x))

    return trainA_image_names, trainB_image_names


def load_A2B(root):
    trainA_root = os.listdir(os.path.join(root + '/SID_long/'))
    trainA_root = [i for i in trainA_root if not i.startswith('.')]
    trainA_root = [i for i in trainA_root if i.startswith('0')]

    trainA_image_names = []
    for i in trainA_root:
        imgs = os.listdir(root + '/SID_long/' + i)
        for x in imgs:
            if x.startswith('h') and x.endswith('.png'):
                trainA_image_names.append(root + '/SID_long/' + i + '/' + x)

    trainB_root = os.path.join(root, 'vimeo')
    trainB_image_names = []
    trainB_path = os.listdir(trainB_root)
    for i in trainB_path:
        trainB_subpath = os.listdir(os.path.join(trainB_root, i))
        for j in trainB_subpath:
            for p in range(7):
                trainB_image_names.append(trainB_root + '/' + i + '/' + j + '/' + 'im' + str(p + 1) + '.png')
    random.shuffle(trainB_image_names)
    return trainA_image_names, trainB_image_names


def create_image_array(image_list, crop_size, do_crop):
    image_array = []
    for image_name in image_list:
        try:
            image = np.array(Image.open(image_name).convert("RGB"))
            if do_crop:
                image = cv2.resize(image, (448, 256))
                cropped_image = random_crop(image, (crop_size, crop_size))
                cropped_image = normalize_negative_one(cropped_image)
            else:
                cropped_image = image
            image_array.append(cropped_image)
        except:
            continue
    image_array = np.array(image_array)
    return image_array


def create_image_array_b2c_sup(root, image_listB, crop_size):
    image_array_A = []
    image_array_B = []

    print(image_listB)
    for img_path in image_listB:
        img_short = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
        img_short = np.float32(img_short)

        image_name = img_path.split('/')[-2]

        l = os.listdir(os.path.join(root, 'SID_long', image_name))
        l = [x for x in l if x.endswith('.png') and x.startswith('h')]

        img_long = cv2.imread(os.path.join(root, 'SID_long', image_name, l[0]), cv2.IMREAD_UNCHANGED)
        img_long = np.float32(img_long)

        img_short = cv2.resize(img_short, (448, 256))
        img_long = cv2.resize(img_long, (448, 256))

        raw_crop, long_crop = random_crop_b2c_sup(img_short, img_long, crop_size)

        raw_crop = normalize_negative_one(raw_crop)
        long_crop = normalize_negative_one(long_crop)

        image_array_A.append(long_crop)
        image_array_B.append(raw_crop)

    image_array_A = np.array(image_array_A)
    image_array_B = np.array(image_array_B)

    return image_array_A, image_array_B


class data_sequence(Sequence):
    def __init__(self, root, image_list_A, image_list_B, crop_size, task, batch_size=1):

        self.batch_size = batch_size
        self.train_A = []
        self.train_B = []

        self.test_A = None
        self.test_B = None

        self.root = root

        self.task = task
        self.crop_size = crop_size

        for image_name in image_list_A:
            if os.path.isfile(image_name):
                self.train_A.append(image_name)
        for image_name in image_list_B:
            if os.path.isfile(image_name):
                self.train_B.append(image_name)

    def __len__(self):
        return int(max(len(self.train_A), len(self.train_B)) / float(self.batch_size))

    def __getitem__(self, idx):
        if idx >= min(len(self.train_A), len(self.train_B)):
            # If all images soon are used for one domain,
            # randomly pick from this domain
            if len(self.train_A) <= len(self.train_B):
                indexes_A = np.random.randint(len(self.train_A), size=self.batch_size)
                batch_A = []
                for i in indexes_A:
                    batch_A.append(self.train_A[i])
                    batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]
            else:
                indexes_B = np.random.randint(len(self.train_B), size=self.batch_size)
                batch_B = []
                for i in indexes_B:
                    batch_B.append(self.train_B[i])
                    batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]

        else:
            batch_A = self.train_A[idx * self.batch_size:(idx + 1) * self.batch_size]
            batch_B = self.train_B[idx * self.batch_size:(idx + 1) * self.batch_size]

        if self.task == 'A2B' or self.task == 'B2C':
            real_images_A = create_image_array(batch_A, self.crop_size, do_crop=True)
            real_images_B = create_image_array(batch_B, self.crop_size, do_crop=True)

        elif self.task == 'B2C_sup':
            real_images_A, real_images_B = create_image_array_b2c_sup(self.root, batch_B, self.crop_size)

        return real_images_A, real_images_B
