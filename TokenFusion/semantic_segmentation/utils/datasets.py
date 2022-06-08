import os
import numpy as np
import cv2
from PIL import Image
from torch.utils.data import Dataset


def line_to_paths_fn_nyudv2(x, input_names):
    return x.decode('utf-8').strip('\n').split('\t')

line_to_paths_fn = {'nyudv2': line_to_paths_fn_nyudv2}


class SegDataset(Dataset):
    """Multi-Modality Segmentation dataset.

    Works with any datasets that contain image
    and any number of 2D-annotations.

    Args:
        data_file (string): Path to the data file with annotations.
        data_dir (string): Directory with all the images.
        line_to_paths_fn (callable): function to convert a line of data_file
            into paths (img_relpath, msk_relpath, ...).
        masks_names (list of strings): keys for each annotation mask
                                        (e.g., 'segm', 'depth').
        transform_trn (callable, optional): Optional transform
            to be applied on a sample during the training stage.
        transform_val (callable, optional): Optional transform
            to be applied on a sample during the validation stage.
        stage (str): initial stage of dataset - either 'train' or 'val'.

    """
    def __init__(self, dataset, data_file, data_dir, input_names, input_mask_idxs,
                 transform_trn=None, transform_val=None, stage='train', ignore_label=None):
        with open(data_file, 'rb') as f:
            datalist = f.readlines()
        self.datalist = [line_to_paths_fn[dataset](l, input_names) for l in datalist]
        self.root_dir = data_dir
        self.transform_trn = transform_trn
        self.transform_val = transform_val
        self.stage = stage
        self.input_names = input_names
        self.input_mask_idxs = input_mask_idxs
        self.ignore_label = ignore_label

    def set_stage(self, stage):
        """Define which set of transformation to use.

        Args:
            stage (str): either 'train' or 'val'

        """
        self.stage = stage

    def __len__(self):
        return len(self.datalist)

    def __getitem__(self, idx):
        idxs = self.input_mask_idxs
        names = [os.path.join(self.root_dir, rpath) for rpath in self.datalist[idx]]
        sample = {}
        for i, key in enumerate(self.input_names):
            sample[key] = self.read_image(names[idxs[i]], key)
        try:
            mask = np.array(Image.open(names[idxs[-1]]))
        except FileNotFoundError:  # for sunrgbd
            path = names[idxs[-1]]
            num_idx = int(path[-10:-4]) + 5050
            path = path[:-10] + '%06d' % num_idx + path[-4:]
            mask = np.array(Image.open(path))
        assert len(mask.shape) == 2, 'Masks must be encoded without colourmap'
        sample['inputs'] = self.input_names
        sample['mask'] = mask
        if self.stage == 'train':
            if self.transform_trn:
                sample = self.transform_trn(sample)
        elif self.stage == 'val':
            if self.transform_val:
                sample = self.transform_val(sample)
        del sample['inputs']
        return sample
    
    @staticmethod
    def read_image_(x, key):
        img = cv2.imread(x)
        if key == 'depth':
            img = cv2.applyColorMap(cv2.convertScaleAbs(255 - img, alpha=1), cv2.COLORMAP_JET)
        return img

    @staticmethod
    def read_image(x, key):
        """Simple image reader

        Args:
            x (str): path to image.

        Returns image as `np.array`.

        """
        img_arr = np.array(Image.open(x))
        if len(img_arr.shape) == 2:  # grayscale
            img_arr = np.tile(img_arr, [3, 1, 1]).transpose(1, 2, 0)
        return img_arr
