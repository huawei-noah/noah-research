# Custom dataset
from PIL import Image
import torch.utils.data as data
import numpy as np
import os
import random

max_v = {'edge_texture': 11355, 'edge_occlusion': 11584, 'depth_euclidean': 11.1, 'depth_zbuffer': 11.1}
min_v = {'edge_texture': 0, 'edge_occlusion': 0, 'depth_euclidean': 0, 'depth_zbuffer': 0}


def line_to_path_fn(x, data_dir):
    path = x.decode('utf-8').strip('\n')
    return os.path.join(data_dir, path)


class DatasetFromFolder(data.Dataset):
    def __init__(self, data_dir, data_file, img_types, transform=None,
                 resize_scale=None, crop_size=None, fliplr=False, is_cls=False):
        super(DatasetFromFolder, self).__init__()
        with open(data_file, 'rb') as f:
            data_list = f.readlines()
        self.data_list = [line_to_path_fn(line, data_dir) for line in data_list]
        self.img_types = img_types
        self.transform = transform
        self.resize_scale = resize_scale
        self.crop_size = crop_size
        self.fliplr = fliplr
        self.is_cls = is_cls

    def __getitem__(self, index):
        # Load Image
        domain_path = self.data_list[index]
        if self.is_cls:
            img_types = self.img_types[:-1]
            cls_target = self.img_types[-1]
        else:
            img_types = self.img_types
        img_paths = [domain_path.replace('{domain}', img_type) for img_type in img_types]
        imgs = [Image.open(img_path) for img_path in img_paths]

        for l in range(len(imgs)):
            img = np.array(imgs[l])
            img_type = img_types[l]
            update = False
            if len(img.shape) == 2:
                img = img[:,:, np.newaxis]
                img = np.concatenate([img] * 3, 2)
                update = True
            if 'depth' in img_type:
                img = np.log(1 + img)
                update = True
            if img_type in max_v:
                img = (img - min_v[img_type]) * 255.0 / (max_v[img_type] - min_v[img_type])
                update = True
            if update:
                imgs[l] = Image.fromarray(img.astype('uint8'))
        
        if self.resize_scale:
            imgs = [img.resize((self.resize_scale, self.resize_scale), Image.BILINEAR) \
                for img in imgs]
        if self.crop_size:
            x = random.randint(0, self.resize_scale - self.crop_size + 1)
            y = random.randint(0, self.resize_scale - self.crop_size + 1)
            imgs = [img.crop((x, y, x + self.crop_size, y + self.crop_size)) for img in imgs]
        if self.fliplr:
            if random.random() < 0.5:
                imgs = [img.transpose(Image.FLIP_LEFT_RIGHT) for img in imgs]
        if self.transform is not None:
            imgs = [self.transform(img) for img in imgs]

        if self.is_cls:
            inputs = imgs
            target = np.load(domain_path.replace('{domain}', cls_target).\
                replace('png', 'npy').replace('scene.npy', 'places.npy'))
            target = np.argmax(target)
        else:
            inputs, target = imgs[:-1], imgs[-1]
            
        return inputs, target

    def __len__(self):
        return len(self.data_list)
