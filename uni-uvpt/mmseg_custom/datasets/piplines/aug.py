# Copyright 2023 Huawei Technologies Co., Ltd
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================
# Modified from: https://github.com/open-mmlab/mmsegmentation/tree/v0.30.0

import copy

import mmcv
import numpy as np
from mmcv.utils import deprecated_api_warning, is_tuple_of
from numpy import random
from pathlib import Path
import cv2
import imgaug.augmenters as iaa

from mmcv.parallel import DataContainer as DC
from mmseg.datasets.builder import PIPELINES
from mmseg.datasets.pipelines import to_tensor
from .GtA_aug import net 
from os.path import join, dirname
from .GtA_aug.utils import *
from .GtA_aug.styleaug import StyleAugmentor


# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
device = torch.device('cpu')


@PIPELINES.register_module()
class DefaultFormatBundleAug(object):
    """Default formatting bundle.

    It simplifies the pipeline of formatting common fields, including "img"
    and "gt_semantic_seg". These fields are formatted as follows.

    - img: (1)transpose, (2)to tensor, (3)to DataContainer (stack=True)
    - gt_semantic_seg: (1)unsqueeze dim-0 (2)to tensor,
                       (3)to DataContainer (stack=True)
    """

    def __call__(self, results):
        """Call function to transform and format common fields in results.

        Args:
            results (dict): Result dict contains the data to convert.

        Returns:
            dict: The result dict contains the data that is formatted with
                default bundle.
        """

        for key in results.keys():
            if key == 'img' or key == 'img_snow' or key == 'img_frost':
                img = results[key]
                if len(img.shape) < 3:
                    img = np.expand_dims(img, -1)
                img = np.ascontiguousarray(img.transpose(2, 0, 1))
                results[key] = DC(to_tensor(img), stack=True)

        if 'gt_semantic_seg' in results:
            # convert to long
            results['gt_semantic_seg'] = DC(
                to_tensor(results['gt_semantic_seg'][None,
                                                     ...].astype(np.int64)),
                stack=True)
        return results

    def __repr__(self):
        return self.__class__.__name__


@PIPELINES.register_module()
class NormalizeAug(object):
    """Normalize the image.

    Added key is "img_norm_cfg".

    Args:
        mean (sequence): Mean values of 3 channels.
        std (sequence): Std values of 3 channels.
        to_rgb (bool): Whether to convert the image from BGR to RGB,
            default is true.
    """

    def __init__(self, mean, std, to_rgb=True):
        self.mean = np.array(mean, dtype=np.float32)
        self.std = np.array(std, dtype=np.float32)
        self.to_rgb = to_rgb

    def __call__(self, results):
        """Call function to normalize images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Normalized results, 'img_norm_cfg' key is added into
                result dict.
        """
        for key in results.keys():
            if key == 'img' or key == 'img_snow' or key == 'img_frost':
                results[key] = mmcv.imnormalize(results[key], self.mean, self.std, self.to_rgb)

        results['img_norm_cfg'] = dict(mean=self.mean, std=self.std, to_rgb=self.to_rgb)

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(mean={self.mean}, std={self.std}, to_rgb=' \
                    f'{self.to_rgb})'
        return repr_str


@PIPELINES.register_module()
class PadAug(object):
    """Pad the image & mask.

    There are two padding modes: (1) pad to a fixed size and (2) pad to the
    minimum size that is divisible by some number.
    Added keys are "pad_shape", "pad_fixed_size", "pad_size_divisor",

    Args:
        size (tuple, optional): Fixed padding size.
        size_divisor (int, optional): The divisor of padded size.
        pad_val (float, optional): Padding value. Default: 0.
        seg_pad_val (float, optional): Padding value of segmentation map.
            Default: 255.
    """

    def __init__(self,
                 size=None,
                 size_divisor=None,
                 pad_val=0,
                 seg_pad_val=255):
        self.size = size
        self.size_divisor = size_divisor
        self.pad_val = pad_val
        self.seg_pad_val = seg_pad_val
        # only one of size and size_divisor should be valid
        assert size is not None or size_divisor is not None
        assert size is None or size_divisor is None

    def _pad_img(self, results):
        """Pad images according to ``self.size``."""

        for key in results.keys():
            if key == 'img' or key == 'img_snow' or key == 'img_frost':
                if self.size is not None:
                    padded_image =  mmcv.impad(results[key], shape=self.size, pad_val=self.pad_val)
                elif self.size_divisor is not None:
                    padded_image = mmcv.impad_to_multiple(results[key], self.size_divisor, pad_val=self.pad_val)
                results[key] = padded_image

        results['pad_shape'] = padded_image.shape
        results['pad_fixed_size'] = self.size
        results['pad_size_divisor'] = self.size_divisor

    def _pad_seg(self, results):
        """Pad masks according to ``results['pad_shape']``."""
        for key in results.get('seg_fields', []):
            results[key] = mmcv.impad(
                results[key],
                shape=results['pad_shape'][:2],
                pad_val=self.seg_pad_val)

    def __call__(self, results):
        """Call function to pad images, masks, semantic segmentation maps.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Updated result dict.
        """

        self._pad_img(results)
        self._pad_seg(results)
        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += f'(size={self.size}, size_divisor={self.size_divisor}, ' \
                    f'pad_val={self.pad_val})'
        return repr_str


@PIPELINES.register_module()
class Adain(object):
    """
    Args:
    """

    def __init__(self, adain=0.3):
        self.adain = adain

        self.vgg = net.vgg
        self.decoder = net.decoder
        self.decoder.eval()
        self.vgg.eval()
        # self.decoder.load_state_dict(torch.load(join(dirname(__file__), 'GtA_aug/checkpoints/decoder.pth')))
        # self.vgg.load_state_dict(torch.load(join(dirname(__file__), 'GtA_aug/checkpoints/vgg_normalised.pth')))
        self.decoder.load_state_dict(torch.load('model/piplines/decoder.pth'))
        self.vgg.load_state_dict(torch.load('model/piplines/vgg_normalised.pth'))
        self.vgg = nn.Sequential(*list(self.vgg.children())[:31])
        self.vgg.to(device)
        self.decoder.to(device)
        self.style_tf = test_transform(512, False)
        self.content_tf = test_transform(512, False)
        style_dir = Path(join(dirname(__file__), 'model/piplines/input/style'))
        self.style_paths = [f for f in style_dir.glob('*')]

    def __call__(self, results):
        """Call function to perform ** on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        content = ((results['img']).copy())[:,:,::-1]# h w c RGB
        content = transforms.ToTensor()(content.copy()) # c h w  RGB
        style_choice = random.randint(0, len(self.style_paths) - 1)

        style = Image.open(str(self.style_paths[style_choice])) # h w c RGB
        style = self.style_tf(style) # c h w RGB

        style = style.to(device).unsqueeze(0)
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():
            output = style_transfer(self.vgg, self.decoder, content, style, alpha=self.adain)
        output = output.cpu().squeeze(0).numpy()

        results['img_adain'] = ((output.transpose((1, 2, 0))*255)[:,:,::-1]).copy() # h w c BRG


        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(adain={self.adain}, ')
        return repr_str

@PIPELINES.register_module()
class Styleaug(object):
    """
    Args:
    """

    def __init__(self):
        self.augmentor = StyleAugmentor(device=device)


    def __call__(self, results):
        """Call function to perform ** on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        content = ((results['img']).copy())[:,:,::-1]# h w c RGB
        content = transforms.ToTensor()(content.copy()) # c h w  RGB
        content = content.to(device).unsqueeze(0)

        with torch.no_grad():
            output = self.augmentor(content)
        output = output.cpu().squeeze(0).numpy()

        results['img_styleaug'] = ((output.transpose((1, 2, 0))*255)[:,:,::-1]).copy() # h w c BRG

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str




@PIPELINES.register_module()
class Imgaug(object):
    """
    Args:

    """
    def __init__(self, imgaug='snow'):
        self.imgaug = imgaug

    def __call__(self, results):
        """Call function to perform XX on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        content = ((results['img']).copy())[:,:,::-1]# h w c RGB
        severity = random.randint(1, 3)
        if self.imgaug == 'rain':
            aug = iaa.Rain(drop_size=(0.5, 0.7))
        elif self.imgaug == 'snow':
            aug = iaa.imgcorruptlike.Snow(severity=severity)
        elif self.imgaug == 'frost':
            aug = iaa.imgcorruptlike.Frost(severity=severity)
        elif self.imgaug == 'cartoon':
            aug = iaa.Cartoon()

        image = aug(image=content)

        if self.imgaug == 'rain':
            results['img_rain'] = ((image)[:,:,::-1]).copy() # h w c BRG
            # cv2.imwrite("raincontent.png", (content)[:,:,::-1])
            # cv2.imwrite("img_rain.png", results['img_rain'])
        elif self.imgaug == 'snow':
            results['img_snow'] = ((image)[:,:,::-1]).copy() # h w c BRG
            # cv2.imwrite("snowcontent.png", (content)[:,:,::-1])
            # cv2.imwrite("img_snow.png", results['img_snow'])
        elif self.imgaug == 'frost':
            results['img_frost'] = ((image)[:,:,::-1]).copy() # h w c BRG
            # cv2.imwrite("frostcontent.png", (content)[:,:,::-1])
            # cv2.imwrite("img_frost.png", results['img_frost'])
        elif self.imgaug == 'cartoon':
            results['img_cartoon'] = ((image)[:,:,::-1]).copy() # h w c BRG
            # cv2.imwrite("cartooncontent.png", (content)[:,:,::-1])
            # cv2.imwrite("img_cartoon.png", results['img_cartoon'])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(imgaug={self.imgaug}, ')
        return repr_str


@PIPELINES.register_module()
class Fda(object):
    """
    Args:

    """
    def __init__(self, fda=('random', 0.005)):
        self.fda = fda[0]
        self.fda_L = fda[1]
        # fda[0] should have path to directory with target images
        tgt_dir = Path(join(dirname(__file__), 'GtA_aug', self.fda))
        self.tgt_paths = [f for f in tgt_dir.glob('*')]

    def __call__(self, results):
        """Call function to perform XX on images.

        Args:
            results (dict): Result dict from loading pipeline.

        Returns:
            dict: Result dict with images distorted.
        """

        source = ((results['img']).copy())[:,:,::-1]# h w c RGB

        if self.fda == 'random':
            target = np.random.uniform(1, 255, source.shape)
        else:
            choice = random.randint(0, len(self.tgt_paths) - 1)
            target = Image.open(self.tgt_paths[choice])
            target = target.resize((512, 512), Image.BICUBIC)

        target = np.asarray(target, np.float32)

        source = source.transpose((2, 0, 1)) # c h w
        target = target.transpose((2, 0, 1)) # c h w

        output = FDA_source_to_target_np(source, target, L=self.fda_L)

        source = source.transpose((1, 2, 0)) # h, w, c RGB
        target = target.transpose((1, 2, 0)) # h, w, c RGB
        output = output.transpose((1, 2, 0)) # h, w, c RGB

        if self.fda == 'random':
            results['img_fda_random'] = ((output)[:,:,::-1]).copy() # h w c BRG
            # cv2.imwrite("source.png", (source)[:,:,::-1])
            # cv2.imwrite("target.png", (target)[:,:,::-1])
            # cv2.imwrite("img_fda_random.png", results['img_fda_random'])
        else:
            results['img_fda'] = ((output)[:,:,::-1]).copy() # h w c BRG
            # cv2.imwrite("source.png", (source)[:,:,::-1])
            # cv2.imwrite("target.png", (target)[:,:,::-1])
            # cv2.imwrite("img_fda.png", results['img_fda'])

        return results

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (f'(fda={self.fda}, ')
        repr_str += (f'(fda_L={self.fda_L}, ')
        return repr_str
