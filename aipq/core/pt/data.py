# Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.         
#                                                                                
# This program is free software; you can redistribute it and/or modify it under  
# the terms of the MIT license.                                                  
#                                                                                
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.                      

from PIL import Image
import cv2
import collections
import os
import random
import torch
import torch.utils.data as data
import torchvision.transforms as transforms
import pandas as pd
import numpy as np
import glob
from PIL import ImageFile, Image
ImageFile.LOAD_TRUNCATED_IMAGES = True

INTER_MODE = {'NEAREST': cv2.INTER_NEAREST, 'BILINEAR': cv2.INTER_LINEAR, 'BICUBIC': cv2.INTER_CUBIC}


def pil_loader(path,color_space='rgb'):
    if color_space == 'y':
        # not implemented yet
        pass
    else:
        # open path as file to avoid ResourceWarning (https://github.com/python-pillow/Pillow/issues/835)
        with open(path, 'rb') as f:
            with Image.open(f) as img:
                return img.convert('RGB')
    return -1


def cv2_loader(fpath,color_space='rgb'):
    if color_space == 'y':
        img = cv2.imread(fpath, cv2.IMREAD_COLOR).astype(np.float32)
        ret_img = np.expand_dims(cv2.cvtColor(img, cv2.COLOR_BGR2YUV)[:,:,0],-1) / 255.
    else:
        img = cv2.imread(fpath, cv2.IMREAD_COLOR).astype(np.float32) / 255.
        ret_img = img[:, :, ::-1]
    return ret_img


def paired_random_crop(imgs, patch_size):
    h, w, _ = np.asarray(imgs[0]).shape

    # randomly choose top and left coordinates for patch
    top = random.randint(0, h - patch_size)
    left = random.randint(0, w - patch_size)

    # crop
    if isinstance(imgs[0], np.ndarray):
        imgs = [img[top:top + patch_size, left:left + patch_size, ...]
                for img in imgs]
    elif isinstance(imgs[0], Image.Image):
        imgs = [img.crop((left, top, left + patch_size, top + patch_size))
                for img in imgs]
    return imgs


def augment(imgs, hflip=True, rotation=True):
    """Augment: horizontal flips OR rotate (0, 90, 180, 270 degrees).
    We use vertical flip and transpose for rotation implementation.
    All the images in the list use the same augmentation.
    Args:
        imgs (list[ndarray] | ndarray): Images to be augmented. If the input
            is an ndarray, it will be transformed to a list.
        hflip (bool): Horizontal flip. Default: True.
        rotation (bool): Ratotation. Default: True.
    Returns:
        list[ndarray] | ndarray: Augmented images and flows. If returned
            results only have one element, just return ndarray.
    """
    hflip = hflip and random.random() < 0.5
    vflip = rotation and random.random() < 0.5
    rot90 = rotation and random.random() < 0.5

    def _augment(img):
        if hflip:  # horizontal
            cv2.flip(img, 1, img)
        if vflip:  # vertical
            cv2.flip(img, 0, img)
        if rot90:
            img = img.transpose(1, 0, 2)
        return img

    def _augmentPIL(img):
        if hflip:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if vflip:
            img = img.transpose(Image.FLIP_TOP_BOTTOM)
        if rot90:
            img = img.transpose(Image.ROTATE_90)
        return img

    if isinstance(imgs[0], np.ndarray):
        fn = _augment
    elif isinstance(imgs[0], Image.Image):
        fn = _augmentPIL

    if not isinstance(imgs, list):
        imgs = [imgs]
    imgs = [fn(img) for img in imgs]
    if len(imgs) == 1:
        imgs = imgs[0]
    return imgs


def img2tensor(imgs, bgr2rgb=True, float32=True):
    """Numpy array to tensor.
    Args:
        imgs (list[ndarray] | ndarray): Input images.
        bgr2rgb (bool): Whether to change bgr to rgb.
        float32 (bool): Whether to change to float32.
    Returns:
        list[tensor] | tensor: Tensor images. If returned results only have
            one element, just return tensor.
    """

    def _totensor(img, bgr2rgb, float32):
        if img.shape[2] == 3 and bgr2rgb:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = torch.from_numpy(img.copy().transpose(2, 0, 1))
        if float32:
            img = img.float()
        return img

    _totensorPIL = transforms.ToTensor()

    if isinstance(imgs, list):  # list of images
        if isinstance(imgs[0], np.ndarray):
            return [_totensor(img, bgr2rgb, float32) for img in imgs]
        if isinstance(imgs[0], Image.Image):
            return [_totensorPIL(img) for img in imgs]
    else:  # single image
        if isinstance(imgs, np.ndarray):
            return _totensor(imgs, bgr2rgb, float32)
        elif isinstance(imgs, Image.Image):
            return _totensorPIL(imgs)


def _is_numpy_image(img):
    return isinstance(img, np.ndarray) and (img.ndim in {2, 3})


def resize(img, size, interpolation='BILINEAR'):
    """Resize the input CV Image to the given size.
    source: https://github.com/YU-Zhiyang/opencv_transforms_torchvision/
    Args:
        img (np.ndarray): Image to be resized.
        size (tuple or int): Desired output size. If size is a sequence like
            (h, w), the output size will be matched to this. If size is an int,
            the smaller edge of the image will be matched to this number maintaing
            the aspect ratio. i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (str, optional): Desired interpolation. Default is ``BILINEAR``
    Returns:
        cv Image: Resized image.
    """
    if not _is_numpy_image(img):
        raise TypeError('img should be CV Image. Got {}'.format(type(img)))
    if not (isinstance(size, int) or (isinstance(size, collections.Iterable) and len(size) == 2)):
        raise TypeError('Got inappropriate size arg: {}'.format(size))

    if isinstance(size, int):
        h, w, c = img.shape
        if (w <= h and w == size) or (h <= w and h == size):
            return img
        if w < h:
            ow = size
            oh = int(size * h / w)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
        else:
            oh = size
            ow = int(size * w / h)
            return cv2.resize(img, dsize=(ow, oh), interpolation=INTER_MODE[interpolation])
    else:
        oh, ow = size
        return cv2.resize(img, dsize=(int(ow), int(oh)), interpolation=INTER_MODE[interpolation])


class TrainTransforms(object):
    def __init__(self, normalize=None, patch_size=64,
                 hflip=True, rotation=True, bgr2rgb=False):
        if normalize is None:
            self.normalize = transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            self.normalize = normalize

        self.patch_size = patch_size
        self.hflip = hflip
        self.rotation = rotation
        self.bgr2rgb = bgr2rgb

    def __call__(self, imgs):
        imgs = paired_random_crop(imgs, self.patch_size)
        imgs = augment(imgs, hflip=self.hflip, rotation=self.rotation)
        imgs = img2tensor(imgs, bgr2rgb=self.bgr2rgb)
        imgs = [self.normalize(img) for img in imgs]
        return imgs


class ValTransforms(object):
    def __init__(self, normalize=None, hflip=False, resize=0, bgr2rgb=False):
        if normalize is None:
            self.normalize = transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            self.normalize = normalize
        self.resize = resize
        self.hflip = hflip

        self.bgr2rgb = bgr2rgb

    def do_hflip(self, img):
        cv2.flip(img, 1, img)
        return img

    def __call__(self, imgs):
        if self.resize > 0:
            imgs = [resize(img, self.resize) for img in imgs]
        if self.hflip:
            imgs = [self.do_hflip(img) for img in imgs]
        imgs = img2tensor(imgs, bgr2rgb=self.bgr2rgb)
        imgs = [self.normalize(img) for img in imgs]
        return imgs


class ValTransformsPIL(object):
    def __init__(self, normalize=None, resize=0):
        if normalize is None:
            self.normalize = transforms.Normalize(
                (0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        else:
            self.normalize = normalize

        self.resize = resize

        if self.resize == 0:
            self.transform = transforms.Compose([
                transforms.ToTensor(),
                normalize,
                ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize(self.resize),
                transforms.ToTensor(),
                normalize,
                ])

    def __call__(self, imgs):
        imgs = [self.transform(img) for img in imgs]
        return imgs



def train_transforms(imgs, patch_size=64):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))

    imgs = paired_random_crop(imgs, patch_size)
    imgs = augment(imgs)

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pylab as plt
    for i, img in enumerate(imgs):
        plt.figure()
        plt.imshow(img, aspect='auto')
        plt.axis('off')
        plt.savefig('%d.png' % i, bbox_inches='tight',
                    pad_inches=0, transparent=True)
    plt.close('all')

    imgs = img2tensor(imgs)

    imgs = [normalize(img) for img in imgs]
    return imgs


def val_transforms(imgs):
    normalize = transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    imgs = img2tensor(imgs)
    imgs = [normalize(img) for img in imgs]
    return imgs


class PIPALsinglewRef(data.Dataset):
    def __init__(self, root, transform,
                 df_path='../data/pipal/train_label.csv',
                 split='train', loader=cv2_loader, sample=True, color_space='rgb'):

        self.root = root

        df = pd.read_csv(df_path)

        if sample and split != 'all':
            np.random.seed(1)
            pairs = df['dist'].astype(str) + '_' + df['ver'].astype(str)

            # 100 out of 116
            n_distortions = len(np.unique(pairs))
            idx = np.random.choice(np.arange(n_distortions),
                                   size=100, replace=False)
            train_distortions = pairs.isin(np.unique(pairs)[idx])

            # 175 out of 200
            n_ref_imgs = len(df['img'].unique())
            idx = np.random.choice(np.arange(n_ref_imgs),
                                   size=175, replace=False)
            train_imgs = df['img'].isin(df['img'].unique()[idx])

            df_train = df.loc[train_imgs & train_distortions]  # 17500 imgs (+ 2800 not used)
            df_val = df.loc[(~train_imgs) & (~train_distortions)]  # 400 imgs (+ 2500 not used)
            df_val_full = df.loc[~train_imgs]
        else:
            train_imgs = df['img'].unique()[:175]
            df_train = df.loc[df['img'].isin(train_imgs)]
            df_val = df.loc[~df['img'].isin(train_imgs)]
            df_val_full = df.loc[~df['img'].isin(train_imgs)]

        if split == 'train':
            self.df = df_train
        elif split == 'val':
            self.df = df_val
        elif split == 'val_full':
            self.df = df_val_full
        elif split == 'all':
            self.df = df

        self.loader = loader
        self.color_space = color_space 
        self.transform = transform

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        fpath_img = os.path.join(self.root, sample['fpath'])
        img = self.loader(fpath_img, self.color_space)

        if 'elo' in sample.keys():
            fpath_ref = os.path.join(self.root, 'Train_Ref', sample.img + '.bmp')
        else:
            fpath_ref = os.path.join(self.root, 'Ref', sample.img + '.bmp')
        ref = self.loader(fpath_ref, self.color_space)

        ref, img = self.transform([ref.copy(), img.copy()])

        if 'elo' in sample.keys():
            return ref, (img, sample['elo'])
        else:
            return ref, img

    def __len__(self):
        return len(self.df)


class PIPALpairwRef(PIPALsinglewRef):
    def __init__(self, root, transform,
                 df_path='../data/pipal/train_label.csv',
                 split='train', loader=cv2_loader, sample=True, color_space='rgb'):

        super(PIPALpairwRef, self).__init__(
            root, transform,
            df_path, split, loader, sample, color_space)

    def __getitem__(self, index):
        # sample
        sample = self.df.iloc[index]
        fpath_img = os.path.join(self.root, sample['fpath'])
        img = self.loader(fpath_img,self.color_space)

        # adversarial from the same ref
        arr = np.argwhere((self.df['img'] == sample.img).values)
        arr = arr[arr != index]
        index_adv = np.random.choice(arr)
        sample_adv = self.df.iloc[index_adv]
        fpath_adv = os.path.join(self.root, sample_adv['fpath'])
        adv = self.loader(fpath_adv,self.color_space)

        # ref image
        fpath_ref = os.path.join(self.root, 'Train_Ref', sample.img + '.bmp')
        ref = self.loader(fpath_ref,self.color_space)

        ref, img, adv = self.transform([ref.copy(), img.copy(), adv.copy()])

        if 'elo' in sample.keys():
            return ref, (img, sample['elo']), (adv, sample_adv['elo'])
        else:
            return ref, img


class KPerRefSampler(data.Sampler):
    '''
    Sample k distorted images for every reference image in a mini-batch

    Adapted from:
    https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/samplers/m_per_class_sampler.py
    '''
    def __init__(self, df, k=1, batch_size=16):
        self.k_per_ref = k
        self.batch_size = batch_size
        
        self.df = df
        self.labels_to_indices = self.get_labels_to_indices(self.df['img'])
        self.labels = list(self.labels_to_indices.keys())

        self.ref_images = len(self.labels)
        self.distortions = len(self.labels_to_indices[self.labels[0]])

        n_samples = len(self.df['img'])
        self.num_iters = (self.distortions // self.k_per_ref)
        self.list_size = self.num_iters * self.ref_images * self.k_per_ref

    def get_labels_to_indices(self, labels):
        labels_to_indices = collections.defaultdict(list)
        for i, label in enumerate(labels):
            labels_to_indices[label].append(i)
        for k, v in labels_to_indices.items():
            labels_to_indices[k] = np.array(v, dtype=np.int)
        return labels_to_indices

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx_list = []

        for label in self.labels:
            np.random.shuffle(self.labels_to_indices[label])

        for i in range(self.num_iters):
            np.random.shuffle(self.labels)
            for label in self.labels:
                start = i * self.k_per_ref
                end = start + self.k_per_ref
                to_select = self.labels_to_indices[label][start:end]
                idx_list.append(to_select)
        idx_list = np.concatenate(np.asarray(idx_list)).flatten()
        self.idx_list = idx_list

        if self.k_per_ref != self.batch_size:
            idx_list = idx_list.reshape(-1, self.batch_size)
            np.random.shuffle(idx_list)
            idx_list = idx_list.flatten()

        return iter(list(idx_list))


class MOSVicinitySampler(data.Sampler):
    '''
    Sample distorted images within the same MOS vicinity in a mini-batch

    Adapted from:
    https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/samplers/m_per_class_sampler.py
    '''
    def __init__(self, df, vicinity=50, batch_size=16, weighting=True):
        self.df = df
        self.batch_size = batch_size
        self.vicinity = vicinity

        self.n_samples = len(self.df['img'])
        self.num_iters = self.n_samples // self.batch_size
        self.list_size = self.num_iters * self.batch_size
        self.weighting = weighting

    def __len__(self):
        return self.list_size

    def __iter__(self):
        anchors = np.random.choice(
            np.arange(self.n_samples), size=self.num_iters, replace=False)

        weights = np.ones(self.n_samples)

        idx_list = []
        for i, anchor in enumerate(anchors):
            idx_list.append(anchor)

            mos_anchor = self.df.iloc[anchor]['elo']
            floor = np.floor(mos_anchor - self.vicinity)
            ceil = np.ceil(mos_anchor + self.vicinity)
            cond = (self.df['elo'].values > floor) & (self.df['elo'].values < ceil)
            cond[anchors] = False
            to_select = np.argwhere(cond).squeeze()
            replace = len(to_select) < (self.batch_size - 1)
            probs = weights[cond]
            probs = probs / probs.sum()
            idx_list.extend(np.random.choice(to_select, size=self.batch_size - 1, replace=replace, p=probs))
            if self.weighting:
                weights[idx_list[-self.batch_size:]] *= 0.1
        return iter(list(idx_list))


class KPerImageSampler(data.Sampler):
    '''
    Return k copies of the same image in a mini-batch

    Adapted from:
    https://github.com/KevinMusgrave/pytorch-metric-learning/blob/master/src/pytorch_metric_learning/samplers/m_per_class_sampler.py
    '''
    def __init__(self, df, k=2, batch_size=16):
        self.df = df
        self.batch_size = batch_size
        self.k_per_img = k

        self.n_samples = len(self.df['img'])
        self.list_size = self.n_samples * self.k_per_img

    def __len__(self):
        return self.list_size

    def __iter__(self):
        idx = np.arange(self.n_samples)
        np.random.shuffle(idx)

        idx_list = []
        for i in idx:
            idx_list.append(i)
            idx_list.append(i)
        return iter(list(idx_list))


class IQAsinglewRef(data.Dataset):
    '''dataloader for IQA datasets'''
    def __init__(self, root, transform,
                 df_path='../data/pipal/train_label.csv',
                 loader=cv2_loader, color_space='rgb'):

        self.root = root

        ext = os.path.splitext(df_path)[-1]
        if ext == '.hdf5':
            df = pd.read_hdf(df_path)
        elif ext == '.csv':
            df = pd.read_csv(df_path, index_col=0)

        self.df = df

        self.loader = loader
        self.color_space = color_space
        self.transform = transform

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        fpath_img = os.path.join(self.root, sample.name)
        img = self.loader(fpath_img,self.color_space)

        fpath_ref = os.path.join(self.root, sample.ref_img)
        ref = self.loader(fpath_ref,self.color_space)

        ref, img = self.transform([ref.copy(), img.copy()])

        return ref, (img, sample['mos'])

    def __len__(self):
        return len(self.df)


class IQApairwRef(data.Dataset):
    '''dataloader for BAPPS'''
    def __init__(self, root, transform,
                 df_path='../data/pipal/train_label.csv',
                 loader=cv2_loader):

        self.root = root

        ext = os.path.splitext(df_path)[-1]
        if ext == '.hdf5':
            df = pd.read_hdf(df_path)
        elif ext == '.csv':
            df = pd.read_csv(df_path)

        self.df = df

        self.loader = loader
        self.transform = transform

    def __getitem__(self, index):
        sample = self.df.iloc[index]
        fpath_p0 = os.path.join(self.root, sample.p0)
        p0 = self.loader(fpath_p0)

        fpath_p1 = os.path.join(self.root, sample.p1)
        p1 = self.loader(fpath_p1)

        fpath_ref = os.path.join(self.root, sample.ref_img)
        ref = self.loader(fpath_ref)

        ref, p0, p1 = self.transform([ref.copy(), p0.copy(), p1.copy()])

        return ref, (p0, p1, sample['judge'])

    def __len__(self):
        return len(self.df)


class IQAsamplepairwRef(IQAsinglewRef):
    def __init__(self, root, transform,
                 df_path='../data/pipal/train_label.csv',
                 loader=cv2_loader, inference=False, color_space='rgb'):

        super(IQAsamplepairwRef, self).__init__(
            root, transform,
            df_path, loader, color_space)

        self.inference = inference

    def __getitem__(self, index):
        # sample
        sample = self.df.iloc[index]
        fpath_img = os.path.join(self.root, sample.name)
        img = self.loader(fpath_img,self.color_space)

        # adversarial from the same ref
        arr = np.argwhere((self.df['img'] == sample.img).values)
        arr = arr[arr != index]
        index_adv = np.random.choice(arr)
        sample_adv = self.df.iloc[index_adv]
        fpath_adv = os.path.join(self.root, sample_adv.name)
        adv = self.loader(fpath_adv,self.color_space)

        # ref image
        fpath_ref = os.path.join(self.root, sample.ref_img)
        ref = self.loader(fpath_ref,self.color_space)

        ref, img, adv = self.transform([ref.copy(), img.copy(), adv.copy()])

        if self.inference:
            return ref, img
        else:
            return ref, (img, sample['mos']), (adv, sample_adv['mos'])





class IQAsingleNoRef(data.Dataset):
    '''dataloader for IQA datasets without reference image'''
    def __init__(self, root, transform,
                 loader=cv2_loader, color_space='rgb'):

        self.root = root

        self.files = glob.glob("{}/*.*".format(root), recursive=False)

        self.loader = loader
        self.color_space = color_space
        self.transform = transform

    def __getitem__(self, index):
        sample = self.files[index]
        fpath, fname = os.path.split(self.files[index])
        img = self.loader(sample,self.color_space)

        img = self.transform([img.copy()])

        return img, fname

    def __len__(self):
        return len(self.files)

