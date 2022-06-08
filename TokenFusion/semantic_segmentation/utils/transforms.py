# 2022.06.08-Changed for implementation of TokenFusion
#            Huawei Technologies Co., Ltd. <foss@huawei.com>
"""RefineNet-LightWeight

RefineNet-LigthWeight PyTorch for non-commercial purposes

Copyright (c) 2018, Vladimir Nekrasov (vladimir.nekrasov@adelaide.edu.au)
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

* Redistributions of source code must retain the above copyright notice, this
  list of conditions and the following disclaimer.

* Redistributions in binary form must reproduce the above copyright notice,
  this list of conditions and the following disclaimer in the documentation
  and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
"""


import cv2
import numpy as np
import torch

# Usual dtypes for common modalities
KEYS_TO_DTYPES = {
    'rgb': torch.float,
    'depth': torch.float,
    'normals': torch.float,
    'mask': torch.long,
}


class Pad(object):
    """Pad image and mask to the desired size.

    Args:
      size (int) : minimum length/width.
      img_val (array) : image padding value.
      msk_val (int) : mask padding value.

    """
    def __init__(self, size, img_val, msk_val):
        assert isinstance(size, int)
        self.size = size
        self.img_val = img_val
        self.msk_val = msk_val

    def __call__(self, sample):
        image = sample['rgb']
        h, w = image.shape[:2]
        h_pad = int(np.clip(((self.size - h) + 1) // 2, 0, 1e6))
        w_pad = int(np.clip(((self.size - w) + 1) // 2, 0, 1e6))
        pad = ((h_pad, h_pad), (w_pad, w_pad))
        for key in sample['inputs']:
            sample[key] = self.transform_input(sample[key], pad)
        sample['mask'] = np.pad(sample['mask'], pad, mode='constant', constant_values=self.msk_val)
        return sample 

    def transform_input(self, input, pad):
        input = np.stack([
            np.pad(input[:, :, c], pad, mode='constant',
            constant_values=self.img_val[c]) for c in range(3)
        ], axis=2)
        return input


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        crop_size (int): Desired output size.

    """
    def __init__(self, crop_size):
        assert isinstance(crop_size, int)
        self.crop_size = crop_size
        if self.crop_size % 2 != 0:
            self.crop_size -= 1

    def __call__(self, sample):
        image = sample['rgb']
        h, w = image.shape[:2]
        new_h = min(h, self.crop_size)
        new_w = min(w, self.crop_size)
        top = np.random.randint(0, h - new_h + 1)
        left = np.random.randint(0, w - new_w + 1)
        for key in sample['inputs']:
            sample[key] = self.transform_input(sample[key], top, new_h, left, new_w)
        sample['mask'] = sample['mask'][top : top + new_h, left : left + new_w]
        return sample

    def transform_input(self, input, top, new_h, left, new_w):
        input = input[top : top + new_h, left : left + new_w]
        return input


class ResizeAndScale(object):
    """Resize shorter/longer side to a given value and randomly scale.

    Args:
        side (int) : shorter / longer side value.
        low_scale (float) : lower scaling bound.
        high_scale (float) : upper scaling bound.
        shorter (bool) : whether to resize shorter / longer side.

    """
    def __init__(self, side, low_scale, high_scale, shorter=True):
        assert isinstance(side, int)
        assert isinstance(low_scale, float)
        assert isinstance(high_scale, float)
        self.side = side
        self.low_scale = low_scale
        self.high_scale = high_scale
        self.shorter = shorter

    def __call__(self, sample):
        image = sample['rgb']
        scale = np.random.uniform(self.low_scale, self.high_scale)
        if self.shorter:
            min_side = min(image.shape[:2])
            if min_side * scale < self.side:
                scale = (self.side * 1. / min_side)
        else:
            max_side = max(image.shape[:2])
            if max_side * scale > self.side:
                scale = (self.side * 1. / max_side)
        inters = {'rgb': cv2.INTER_CUBIC, 'depth': cv2.INTER_NEAREST}
        for key in sample['inputs']:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], scale, inter)
        sample['mask'] = cv2.resize(sample['mask'], None, fx=scale, fy=scale,
                                    interpolation=cv2.INTER_NEAREST)
        return sample

    def transform_input(self, input, scale, inter):
        input = cv2.resize(input, None, fx=scale, fy=scale, interpolation=inter)
        return input


class CropAlignToMask(object):
    """Crop inputs to the size of the mask."""
    def __call__(self, sample):
        mask_h, mask_w = sample['mask'].shape[:2]
        for key in sample['inputs']:
            sample[key] = self.transform_input(sample[key], mask_h, mask_w)
        return sample

    def transform_input(self, input, mask_h, mask_w):
        input_h, input_w = input.shape[:2]
        if (input_h, input_w) == (mask_h, mask_w):
            return input
        h, w = (input_h - mask_h) // 2, (input_w - mask_w) // 2
        del_h, del_w = (input_h - mask_h) % 2, (input_w - mask_w) % 2
        input = input[h: input_h - h - del_h, w: input_w - w - del_w]
        assert input.shape[:2] == (mask_h, mask_w)
        return input


class ResizeAlignToMask(object):
    """Resize inputs to the size of the mask."""
    def __call__(self, sample):
        mask_h, mask_w = sample['mask'].shape[:2]
        assert mask_h == mask_w
        inters = {'rgb': cv2.INTER_CUBIC, 'depth': cv2.INTER_NEAREST}
        for key in sample['inputs']:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], mask_h, inter)
        return sample

    def transform_input(self, input, mask_h, inter):
        input_h, input_w = input.shape[:2]
        assert input_h == input_w
        scale = mask_h / input_h
        input = cv2.resize(input, None, fx=scale, fy=scale, interpolation=inter)
        return input


class ResizeInputs(object):
    def __init__(self, size):
        self.size = size

    def __call__(self, sample):
        # sample['rgb'] = sample['rgb'].numpy()
        if self.size is None:
            return sample
        size = sample['rgb'].shape[0]
        scale = self.size / size
        # print(sample['rgb'].shape, type(sample['rgb']))
        inters = {'rgb': cv2.INTER_CUBIC, 'depth': cv2.INTER_NEAREST}
        for key in sample['inputs']:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], scale, inter)
        return sample

    def transform_input(self, input, scale, inter):
        input = cv2.resize(input, None, fx=scale, fy=scale, interpolation=inter)
        return input


class ResizeInputsScale(object):
    def __init__(self, scale):
        self.scale = scale

    def __call__(self, sample):
        if self.scale is None:
            return sample
        inters = {'rgb': cv2.INTER_CUBIC, 'depth': cv2.INTER_NEAREST}
        for key in sample['inputs']:
            inter = inters[key] if key in inters else cv2.INTER_CUBIC
            sample[key] = self.transform_input(sample[key], self.scale, inter)
        return sample

    def transform_input(self, input, scale, inter):
        input = cv2.resize(input, None, fx=scale, fy=scale, interpolation=inter)
        return input


class RandomMirror(object):
    """Randomly flip the image and the mask"""
    def __call__(self, sample):
        do_mirror = np.random.randint(2)
        if do_mirror:
            for key in sample['inputs']:
                sample[key] = cv2.flip(sample[key], 1)
            sample['mask'] = cv2.flip(sample['mask'], 1)
        return sample


class Normalise(object):
    """Normalise a tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalise each channel of the torch.*Tensor, i.e.
    channel = (scale * channel - mean) / std

    Args:
        scale (float): Scaling constant.
        mean (sequence): Sequence of means for R,G,B channels respecitvely.
        std (sequence): Sequence of standard deviations for R,G,B channels
            respecitvely.
        depth_scale (float): Depth divisor for depth annotations.

    """
    def __init__(self, scale, mean, std, depth_scale=1.):
        self.scale = scale
        self.mean = mean
        self.std = std
        self.depth_scale = depth_scale

    def __call__(self, sample):
        for key in sample['inputs']:
            if key == 'depth':
                continue
            sample[key] = (self.scale * sample[key] - self.mean) / self.std
        if 'depth' in sample:
            # sample['depth'] = self.scale * sample['depth']
            # sample['depth'] = (self.scale * sample['depth'] - self.mean) / self.std
            if self.depth_scale > 0:
                sample['depth'] = self.depth_scale * sample['depth']
            elif self.depth_scale == -1:  # taskonomy
                # sample['depth'] = np.log(1 + sample['depth']) / np.log(2.** 16.0)
                sample['depth'] = np.log(1 + sample['depth'])
            elif self.depth_scale == -2:  # sunrgbd
                depth = sample['depth']
                sample['depth'] = (depth - depth.min()) * 255.0 / (depth.max() - depth.min())
        return sample


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        # swap color axis because
        # numpy image: H x W x C
        # torch image: C X H X W
        for key in sample['inputs']:
            sample[key] = torch.from_numpy(
                sample[key].transpose((2, 0, 1))
            ).to(KEYS_TO_DTYPES[key] if key in KEYS_TO_DTYPES else KEYS_TO_DTYPES['rgb'])
        sample['mask'] = torch.from_numpy(sample['mask']).to(KEYS_TO_DTYPES['mask'])
        return sample


def make_list(x):
    """Returns the given input as a list."""
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]

