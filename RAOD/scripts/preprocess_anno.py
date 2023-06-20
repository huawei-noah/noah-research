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

import io
import os
import json
import argparse
import numpy as np
from glob import glob
from tqdm import tqdm


# classname_to_id = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Tram': 4, 'Tricycle': 5, 'Truck': 6}
classname_to_id = {'Pedestrian': 1, 'Car': 2, 'Cyclist': 3, 'Tram': 4, 'Truck': 5}
class Lableme2CoCo:
    def __init__(self, img_postfix):
        self.images = []
        self.annotations = []
        self.categories = []
        self.img_id = 0
        self.ann_id = 0
        self.img_postfix = img_postfix

    def save_coco_json(self, instance, save_path):
        print(f'save to : {save_path}')
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with io.open(save_path, 'w', encoding='utf-8') as outfile:
            my_json_str = json.dumps(instance, ensure_ascii=False, indent=1)
            if isinstance(my_json_str, str):
                my_json_str = my_json_str.encode('utf-8').decode('utf-8')
            outfile.write(my_json_str)

    def to_coco(self, json_path_list):
        self._init_categories()
        for json_path in tqdm(json_path_list):
            obj = self.read_jsonfile(json_path)
            self.images.append(self._image(obj, json_path))
            shapes = obj['shapes']
            for shape in shapes:
                if shape['label'] not in classname_to_id.keys():
                    print(shape['label'] )
                    continue
                annotation = self._annotation(shape)
                if annotation['bbox'][2] < 16 or annotation['bbox'][3] < 16:
                    continue
                self.annotations.append(annotation)
                self.ann_id += 1
            self.img_id += 1
        instance = {}
        instance['info'] = ''
        instance['license'] = ['']
        instance['images'] = self.images
        instance['annotations'] = self.annotations
        instance['categories'] = self.categories
        return instance

    # Categories
    def _init_categories(self):
        for k, v in classname_to_id.items():
            category = {}
            category['id'] = v
            category['name'] = k
            self.categories.append(category)

    # COCO Image
    def _image(self, obj, path):
        image = {}
        image['height'] = 1280
        image['width'] = 1280
        image['id'] = self.img_id
        image['file_name'] = os.path.basename(path).replace('.json', self.img_postfix)
        return image

    def _annotation(self, shape):
        label = shape['label']
        points = shape['points']
        annotation = {}
        annotation['id'] = self.ann_id
        annotation['image_id'] = self.img_id
        annotation['category_id'] = int(classname_to_id[label])
        annotation['segmentation'] = [np.asarray(points).flatten().tolist()]
        annotation['bbox'] = self._get_box(points)
        annotation['iscrowd'] = 0
        annotation['area'] = 1.0
        return annotation

    def read_jsonfile(self, path):
        with open(path, 'r') as f:
            return json.load(f)

    def _get_box(self, points):
        min_x = min_y = np.inf
        max_x = max_y = 0
        for x, y in points:
            min_x = min(min_x, x)
            min_y = min(min_y, y)
            max_x = max(max_x, x)
            max_y = max(max_y, y)
            
        min_x = int(min_x * (1280/2880))
        min_y = int(min_y * (1280/1856))
        max_x = int(max_x * (1280/2880))
        max_y = int(max_y * (1280/1856))

        return [min_x, min_y, max_x - min_x, max_y - min_y]


def main(args):
    path = os.path.realpath(args['path']).rstrip('/')
    assert os.path.isdir(path), f'Invalid path < {path} >!'

    files = sorted(glob(os.path.join(path, '*.json')))
    print(f'{len(files)} annotations found')

    l2c = Lableme2CoCo(img_postfix='.npy.gz')
    instance = l2c.to_coco(files)
    l2c.save_coco_json(instance, path + '.json')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, default='/home/data')
    args = parser.parse_args()
    args = args.__dict__
    main(args)
