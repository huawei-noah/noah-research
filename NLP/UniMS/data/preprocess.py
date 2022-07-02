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

import os
from os.path import join
from tqdm import tqdm
from toolz import curry
# import json
import multiprocessing
from multiprocessing import Pool
from rouge import Rouge

import torch
from PIL import Image

import clip
from utils import get_msmo_filelist, parse_article, transform, get_tokenizer


@curry
def get_image_caption_filelist_thread(source_path, data_path):
    images = os.listdir(join(source_path, data_path, 'img'))
    image_list = [join(data_path, 'img', image) for image in images]
    captions = os.listdir(join(source_path, data_path, 'caption'))
    caption_list = [join(data_path, 'caption', caption) for caption in captions]
    return image_list, caption_list


def get_image_caption_filelist(source_path, split):
    if split == 'train':
        # train
        image_list = []
        caption_list = []
        path = [f'data{i + 1}' for i in range(20)]
        with tqdm(total=20) as pbar:
            with Pool(processes=multiprocessing.cpu_count()) as pool:
                for images, captions in pool.imap_unordered(
                        get_image_caption_filelist_thread(source_path), path):
                    image_list += images
                    caption_list += captions
                    pbar.update(1)
    else:
        # valid & test
        image_list, caption_list = get_image_caption_filelist_thread(source_path, f'{split}_data')

    image_list = sorted(image_list, key=lambda k: k.split('/')[-1].split('.')[0].split('_')[0])
    caption_list = sorted(caption_list, key=lambda k: k.split('/')[-1].split('.')[0].split('_')[0])

    return image_list, caption_list


def get_body_filelist(source_path, split):
    if split == 'train':
        # train
        filelist = []
        for i in range(20):
            data_list = get_msmo_filelist(join(source_path, f'data{i + 1}'))
            filelist += [join(f'data{i + 1}', d) for d in data_list]
    else:
        # valid & test
        data_list = get_msmo_filelist(join(source_path, f'{split}_data'))
        filelist = [join(f'{split}_data', d) for d in data_list]

    filelist = sorted(filelist, key=lambda k: k.split('/')[-1].split('.')[0])
    return filelist


def make_article_image_caption_link(source_path, split):
    print("Start Get Files Data.")

    articles = get_body_filelist(source_path, split)
    images, captions = get_image_caption_filelist(source_path, split)

    links = {}
    image_i = 0
    caption_i = 0

    with tqdm(total=len(articles)) as pbar:
        for article in articles:
            article_id = article.split('/')[-1].split('.')[0]
            # get article images and captions
            article_images = []
            article_captions = []

            # get images
            while image_i < len(images):
                image_id = images[image_i].split('/')[-1].split('.')[0].split('_')[0]
                if image_id == article_id:
                    article_images.append(images[image_i])
                    image_i += 1
                else:
                    break

            # get captions
            while caption_i < len(captions):
                caption_id = captions[caption_i].split('/')[-1].split('.')[0].split('_')[0]
                if caption_id == article_id:
                    article_captions.append(captions[caption_i])
                    caption_i += 1
                else:
                    break

            # insert article links
            links[article] = {'images': article_images, 'captions': article_captions}
            pbar.update(1)

    assert len(articles) == len(links)
    assert sum([len(links[k]['images']) for k in links]) == len(images)
    assert sum([len(links[k]['captions']) for k in links]) == len(captions)

    print("Get Files Data End.")
    return links


@curry
def check_file_thread(source_path, input_resolution, item):
    tokenizer = get_tokenizer("BART")
    image_preprocess = transform(input_resolution)

    filename = item[0]
    images = item[1]['images']
    check_images = []
    captions = item[1]['captions']
    check_captions = []

    # Check Images
    if len(images) == 0:
        return None
    else:
        for i in range(len(images)):
            try:
                image = image_preprocess(Image.open(join(source_path, images[i])))
                check_images.append(images[i])
                check_captions.append(captions[i])
            except:
                continue
        if len(check_images) == 0:
            return None

    # Check Article & References
    title, body, references = parse_article(source_path, filename)
    if body == [] or references == []:
        return None
    else:
        body_len = len(tokenizer(' '.join(body))['input_ids'])
        ref_len = len(tokenizer(' '.join(references))['input_ids'])
        if body_len <= ref_len:
            return None

    data = {}
    data[filename] = {
        'images': check_images,
        'captions': check_captions,
    }
    return data


def check_file(source_path, file_data, input_resolution=224):
    print("Start Check Files Data.")

    checked_file_data = {}
    items = file_data.items()
    # items = list(file_data.items())[:10]
    with tqdm(total=len(file_data)) as pbar:
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            for data in pool.imap_unordered(check_file_thread(source_path, input_resolution), items):
                if data is not None:
                    checked_file_data.update(data)
                pbar.update(1)

    print("Check Files Data End.")
    print(
        f"Raw Files: {len(file_data)}, After Checked: {len(checked_file_data)}, Filtered: {len(file_data) - len(checked_file_data)}.")
    return checked_file_data


@curry
def make_extract_pseudo_label_thread(source_path, item, max_sent_num=50, max_sent_len=75):
    tokenizer = get_tokenizer("BART")
    ROUGE = Rouge()

    filename = item[0]
    title, body, references = parse_article(source_path, filename)
    body = body[:max_sent_num]

    # greedy pseudo label: sent_orders, max_score_sent, over_length_index
    pseudo_label = {}
    sent_orders = []
    rouge_scores = []

    body_now = ''
    check_length = True
    check_max = True
    ori_sent_orders = list(range(len(body)))
    for _ in range(len(body)):
        preds = [body_now + body[order] for order in ori_sent_orders]
        refs = [' '.join(references) for _ in range(len(ori_sent_orders))]
        scores = ROUGE.get_scores(preds, refs)
        order_orders = sorted(
            range(len(ori_sent_orders)), key=lambda k: scores[k]['rouge-l']['f'], reverse=True)

        order = order_orders[0]
        index = ori_sent_orders.pop(order)

        if check_max and rouge_scores != [] and scores[
            order]['rouge-l']['f'] < rouge_scores[-1]['rouge-l']['f']:
            pseudo_label['max_score_sent'] = sent_orders[-1]
            check_max = False

        sent_orders.append(index)
        rouge_scores.append(scores[order])
        body_now += body[index]

        if check_length:
            body_len = len(tokenizer(body_now)['input_ids'])
            ref_len = len(tokenizer(' '.join(references))['input_ids'])
            if body_len >= ref_len:
                pseudo_label['over_length_index'] = index
                check_length = False

    if 'over_length_index' not in pseudo_label:
        pseudo_label['over_length_index'] = sent_orders[-1]
    if 'max_score_sent' not in pseudo_label:
        pseudo_label['max_score_sent'] = sent_orders[-1]
    pseudo_label['sent_orders'] = sent_orders
    # pseudo_label['scores'] = rouge_scores

    item = {filename: item[1]}
    item[filename]["sentence_pseudo_labels"] = pseudo_label
    return item


def make_extract_pseudo_label(source_path, file_data):
    print('Start Make Extract Pseudo Label.')
    data = {}
    with tqdm(total=len(file_data)) as pbar:
        with Pool(processes=multiprocessing.cpu_count()) as pool:
            for item in pool.imap_unordered(make_extract_pseudo_label_thread(source_path), file_data.items()):
                # for item in file_data.items():
                #     item = make_extract_pseudo_label_thread(source_path, item)
                if item is not None:
                    data.update(item)
                pbar.update(1)

    # print(data)
    print('Make Extract Pseudo Label End.')
    print(
        f"Raw Files: {len(file_data)}, After Make Extract Pseudo Label: {len(data)}, Filtered: {len(file_data) - len(data)}.")
    return data


@curry
def make_image_pseudo_label_thread(source_path, model, image_preprocessor, item):
    filename = item[0]
    images = item[1]['images']
    _, _, references = parse_article(source_path, filename)
    references = " ".join(references)

    preprocess_images = []
    for image_filepath in images:
        image = image_preprocessor(Image.open(join(source_path, image_filepath)))
        preprocess_images.append(image)
        if len(preprocess_images) >= 10:
            break

    if torch.cuda.is_available():
        images = torch.stack(preprocess_images).cuda()
        tokenized_texts = clip.tokenize(references).cuda()
    else:
        images = torch.stack(preprocess_images)
        tokenized_texts = clip.tokenize(references)

    _, logits = model(images, tokenized_texts)

    item = {filename: item[1]}
    assert len(logits[0]) == len(preprocess_images)
    item[filename]["image_ranking_scores"] = logits[0].cpu().detach().numpy().tolist()

    return item


def make_image_pseudo_label(source_path, file_data):
    clip_path = "/home/zhangzhengkun/EFS-HK-20/zhangzhengkun/Dataset/PTM/clip-models/ViT-B-32.pt"
    model, image_preprocessor = clip.load(
        clip_path,
        device="cuda" if torch.cuda.is_available() else "cpu",
        jit=False
    )
    func = make_image_pseudo_label_thread(source_path, model, image_preprocessor)

    print('Start Make Image Pseudo Label.')
    data = {}
    with tqdm(total=len(file_data)) as pbar:
        for item in file_data.items():
            item = func(item)
            if item is not None:
                data.update(item)
            pbar.update(1)

    # print(data)
    print('Make Image Pseudo Label End.')
    print(
        f"Raw Files: {len(file_data)}, After Make Image Pseudo Label: {len(data)}, Filtered: {len(file_data) - len(data)}.")
    return data
