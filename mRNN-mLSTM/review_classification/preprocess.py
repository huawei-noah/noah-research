#Copyright (C) 2020. Huawei Technologies Co., Ltd. All rights reserved.

#This program is free software; you can redistribute it and/or modify it under
# the terms of the BSD 3-Clause License.

#This program is distributed in the hope that it will be useful, but WITHOUT
# ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
# FITNESS FOR A PARTICULAR PURPOSE. See the BSD 3-Clause License for
# more details.

import os
import json
import argparse
import numpy as np

from gensim.models import word2vec

# initialization
PARSER = argparse.ArgumentParser()
PARSER.add_argument("--file_path", type=str,
                    default="../data/review_classification/review.json",
                    help="Path to raw data file.")
PARSER.add_argument("--corpus_path", type=str, default="corpus.txt",
                    help="Path to save the corpus.")
PARSER.add_argument("--save_path", type=str,
                    default="../data/review_classification/data.json",
                    help="Path to save the preprocessed dataset.")
PARSER.add_argument("--vec_size", type=int, default=16,
                    help="Dimensionality of the feature vectors.")
PARSER.add_argument("--min_count", type=int, default=1,
                    help="The lowest frequency of words in the vocabulary.")
PARSER.add_argument("--window", type=int, default=16,
                    help="The maximum distance between the current and "
                         "predicted word within a sentence.")
PARSER.add_argument("--algorithm", type=int, default=1,
                    help="The training algorithm of Word2Vec."
                         "1 means skip-gram while 0 means CBOW.")
FLAGS = PARSER.parse_args()


def text_clean(text):
    """Remove Punctuations.
    """
    new_text = text.lower().replace(', ', ' ').replace('. ', ' ').\
        replace('(', '').replace('  ', ' ').replace('  ', ' ').\
        replace(')', '').replace('[', '').replace(']', '').\
        replace('"', '').replace('? ', ' ').replace('¿', '').\
        replace('-', '').replace('. ', ' ').replace('\t', '').\
        replace('\\', '')
    return new_text


def create_corpus():
    """Create corpus.
    """
    assert os.path.exists(FLAGS.file_path)
    with open(FLAGS.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    fo = open(FLAGS.corpus_path, 'w')
    for paper in data['paper']:
        reviews = paper['review']
        for review in reviews:
            if review['lan'] not in ['es']:
                continue
            text = text_clean(review['text']) + '\n'
            fo.write(text)
        fo.write('\n')
    fo.close()
    print('Saved in ', FLAGS.corpus_path)


def train_model():
    """Train Word2Vec mdoel.
    """
    corpus_path = FLAGS.corpus_path
    vec_size = FLAGS.vec_size
    model_min_count = FLAGS.min_count
    model_window = FLAGS.window
    model_sg = FLAGS.algorithm
    sentences = word2vec.Text8Corpus(corpus_path)
    model = word2vec.Word2Vec(sentences, size=vec_size,
                              min_count=model_min_count,
                              window=model_window, sg=model_sg)
    print('# Dictionary = ', len(model.wv.vocab))
    return model


def make_data(model):
    """make data with the trained Word2Vec model.
    """
    def sent2mat(sentence):
        # Encode a sentence into a matrix
        words = sentence.split(' ')
        word_list = []
        for word in words:
            if word == '':
                continue
            word_list.append(list(model.wv[word]))

        return word_list

    assert os.path.exists(FLAGS.file_path)
    with open(FLAGS.file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)

    dataset = []
    labels = []
    for paper in data['paper']:
        reviews = paper['review']
        for review in reviews:
            if review['lan'] not in ['es']:
                continue

            text = text_clean(review['text']).strip()
            if len(text) == 0:
                continue

            text_mat = sent2mat(text)
            dataset.append(text_mat)
            labels.append(int(review['evaluation'])+2)

    with open(FLAGS.save_path, 'w') as fo:
        json.dump({"data": dataset, "label": labels}, fo, cls=JSONEncoder)
    print('Saved in ', FLAGS.save_path)


class JSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        else:
            return super(JSONEncoder, self).default(obj)

if __name__ == "__main__":
    create_corpus()
    encoder = train_model()
    make_data(encoder)
