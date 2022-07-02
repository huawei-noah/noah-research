# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from os.path import join
import logging

from PIL import Image
import torch
from torch.utils.data import Dataset
import cv2

from utils import parse_article, has_meaning

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class MSMODataset(Dataset):
    def __init__(
            self,
            data,
            dataset_path,
            image_preprocess,
            max_sent_num,
            max_image_num,
            image_references,
    ):
        self.data = data
        self.dataset_path = dataset_path
        self.image_preprocess = image_preprocess
        self.max_sent_num = max_sent_num
        self.max_image_num = max_image_num
        self.filelist = list(self.data.keys())
        self.image_references = image_references

    def __len__(self):
        return len(self.filelist)

    def __getitem__(self, item_i):
        filepath = self.filelist[item_i]
        # Get Document (title body summary(Reference))
        title, text_sentences, refers = parse_article(self.dataset_path, filepath, lower=False)

        # Get Sentecne Pseudo Label & Make Sentence Reference
        sentence_label = self.data[filepath]['sentence_pseudo_labels']['sent_orders']
        sentence_label = [sl for sl in sentence_label if sl < self.max_sent_num]
        max_score_sent = self.data[filepath]['sentence_pseudo_labels']['max_score_sent']
        references = {
            'refers': refers,
            'sentence_label': sentence_label,
            'indexes': torch.zeros(self.max_sent_num, dtype=torch.long) + item_i,
            'target': torch.zeros(self.max_sent_num).bool(),
            'labels': torch.zeros(self.max_sent_num),
        }
        for sent_order in sentence_label:
            references['target'][sent_order] = True
            references['labels'][sent_order] = 1
            if sent_order == max_score_sent:
                break

        # Get Images with Label & Caption & Reference
        raw_images = []
        cv_raw_images = []
        images = []
        captions = []
        image_references = {
            'indexes': torch.zeros(self.max_image_num, dtype=torch.long) + item_i,
            'target': torch.zeros(self.max_image_num).bool(),
            'labels': torch.tensor(self.data[filepath]['image_ranking_scores']),
        }
        if self.image_preprocess:
            image_filelist = self.data[filepath]["images"]
            for image_filepath in image_filelist:
                try:
                    cv_raw_image = cv2.imread(join(self.dataset_path, image_filepath))
                    raw_image = Image.open(join(self.dataset_path, image_filepath))
                    image = self.image_preprocess(raw_image)
                except:
                    continue

                try:
                    with open(join(self.dataset_path, image_filepath.replace(
                                    'img/', 'caption/').replace('.jpg', '.caption')), 'r') as file:
                        caption = file.read().strip()
                    if has_meaning(caption):
                        captions.append(caption)
                    else:
                        captions.append('None')
                except:
                    captions.append('None')

                if self.image_references is not None:
                    if filepath in self.image_references and image_filepath.split('/')[-1] in \
                            self.image_references[filepath]:
                        image_references['target'][len(images)] = True
                image_references['labels'][len(images)] = self.data[
                    filepath]['image_ranking_scores'][len(images)]
                cv_raw_images.append(cv_raw_image)
                raw_images.append(raw_image)
                images.append(image)
                if len(images) >= self.max_image_num:
                    break

            images = torch.stack(images)
        else:
            images = None

        return {
            'filename': filepath,
            'article': text_sentences[: self.max_sent_num],
            'reference': references,
            'raw_image': raw_images,
            'cv_raw_image': cv_raw_images,
            'image': images,
            'image_reference': image_references,
            'caption': captions,
        }
