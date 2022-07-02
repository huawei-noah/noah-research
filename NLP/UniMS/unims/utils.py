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


import logging
from os.path import join
import re
import cv2
import numpy as np
from spacy import displacy

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def parse_article(dataset_path, filename, lower=True):
    with open(join(dataset_path, filename), 'r', encoding='utf8') as file:
        content = file.readlines()
    iftitle, ifbody = False, False
    title = []
    body = []
    references = []

    for line in content:
        line = line.strip()
        if not has_meaning(line):
            continue
        elif line == '@title':
            iftitle = True
        elif line == '@body':
            ifbody = True
            iftitle = False
        elif line == '@summary':
            ifbody = False
        else:
            line = fix_missing_period(line)
            if iftitle:
                title.append(line.lower() if lower else line)
            elif ifbody:
                body.append(line.lower() if lower else line)
            else:
                references.append(line.lower() if lower else line)
    return title, body, references


dm_single_close_quote = "\u2019"  # unicode
dm_double_close_quote = "\u201d"
# acceptable ways to end a sentence
END_TOKENS = [
    ".",
    "!",
    "?",
    "...",
    "'",
    "`",
    '"',
    dm_single_close_quote,
    dm_double_close_quote,
    ")",
]


def fix_missing_period(line):
    """Adds a period to a line that is missing a period"""
    if "@highlight" in line:
        return line
    if line == "":
        return line
    if line[-1] in END_TOKENS:
        return line
    return line + " ."


def has_meaning(s):
    h = False
    for c in s:
        h = h or c.isalnum()
    return h


def n_grams(tokens, n):
    l = len(tokens)
    return [tuple(tokens[i:i + n]) for i in range(l) if i + n < l]


def cal_novel(summary, gold, source, summary_ngram_novel, gold_ngram_novel):
    summary = summary.replace('<q>', ' ')
    summary = re.sub(r' +', ' ', summary).strip()
    gold = gold.replace('<q>', ' ')
    gold = re.sub(r' +', ' ', gold).strip()
    source = source.replace(' ##', '')
    source = source.replace('[CLS]', ' ').replace('[SEP]', ' ').replace('[PAD]', ' ')
    source = re.sub(r' +', ' ', source).strip()

    for n in summary_ngram_novel.keys():
        summary_grams = set(n_grams(summary.split(), n))
        gold_grams = set(n_grams(gold.split(), n))
        source_grams = set(n_grams(source.split(), n))

        summary_joint = summary_grams.intersection(source_grams)
        summary_novel = summary_grams - summary_joint
        summary_ngram_novel[n][0] += 1.0 * len(summary_novel)
        summary_ngram_novel[n][1] += len(summary_grams)
        summary_ngram_novel[n][2] += 1.0 * len(summary_novel) / (len(summary.split()) + 1e-6)

        gold_joint = gold_grams.intersection(source_grams)
        gold_novel = gold_grams - gold_joint
        # gold_ngram_novel[n][0] += 1.0 * len(gold_novel)
        # gold_ngram_novel[n][1] += len(gold_grams)
        novel = gold_novel.intersection(summary_novel)
        gold_ngram_novel[n][0] += 1.0 * len(novel)
        gold_ngram_novel[n][1] += len(gold_novel)
        gold_ngram_novel[n][2] += 1.0 * len(gold_novel) / (len(gold.split()) + 1e-6)


def save_CAM(cam, img, output_path):
    height, width, _ = img.shape
    # generate class activation mapping for the top1 prediction
    size_upsample = (256, 256)
    cam = cam - np.min(cam)
    cam_img = cam / np.max(cam)
    cam_img = np.uint8(255 * cam_img)
    cam_img = cv2.resize(cam_img, size_upsample)

    # render the CAM and output
    heatmap = cv2.applyColorMap(cam_img, cv2.COLORMAP_JET)
    img = cv2.resize(img, size_upsample)
    result = heatmap * 0.3 + img * 0.5
    cv2.imwrite(output_path, result)


def gamma_correction(score, gamma):
    return np.sign(score) * np.power(np.abs(score), gamma)


def save_render(tokens, scores, save_path):
    colors = {}
    ents = []

    ii = 0
    for token, score in zip(tokens, scores):
        score = gamma_correction(score, 1.0)
        if score >= 0:
            r = str(int(255))
            g = str(int(255 * (1 - score)))
            b = str(int(255 * (1 - score)))
        else:
            r = str(int(255 * (1 + score)))
            g = str(int(255 * (1 + score)))
            b = str(int(255))

        # TODO: Add more color schemes from: https://colorbrewer2.org/#type=diverging&scheme=RdBu&n=5
        red = r
        green = g
        blue = b
        score = round(score, ndigits=3)
        colors[str(score)] = '#%02x%02x%02x' % (int(red), int(green), int(blue))

        ff = ii + len(token)
        ent = {
            'start': ii,
            'end': ff,
            'label': str(score),
        }
        ents.append(ent)
        ii = ff

    to_render = {
        'text': ''.join([token for token in tokens]),
        'ents': ents,
    }

    template = """
    <mark class="entity" style="background: {bg}; padding: 0.15em 0.3em; margin: 0 0.2em; line-height: 2.2;
    border-radius: 0.25em; box-decoration-break: clone; -webkit-box-decoration-break: clone">
        {text}
    </mark>
    """

    html = displacy.render(
        to_render,
        style='ent',
        manual=True,
        jupyter=False,
        options={'template': template,
                 'colors': colors,
                 }
    )

    with open(save_path, 'w') as file:
        file.write(html)
