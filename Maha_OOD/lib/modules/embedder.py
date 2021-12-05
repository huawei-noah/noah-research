# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

from os import PathLike
from pathlib import Path
import pickle
from typing import Optional

import bcolz
import numpy as np
import torch
import torch.nn as nn

from lib.data_utils import Vocab


GLOVE_STD = 0.3836


def preprocess_glove_vectors(path_to_file: PathLike):
    path_to_file = Path(path_to_file)
    words = []
    idx = 0
    word2idx = {}
    store_dir = path_to_file.parent / path_to_file.stem
    store_dir.mkdir(exist_ok=False, parents=True)
    vectors = bcolz.carray(np.zeros(1), rootdir=str(store_dir / f"{path_to_file.stem}.dat"), mode='w')

    dim = None
    with path_to_file.open('rb') as fp:
        for l in fp:
            line = l.decode().split()
            word = line[0]
            words.append(word)
            word2idx[word] = idx
            idx += 1
            vect = np.array(line[1:]).astype(np.float)
            vectors.append(vect)
            if dim is None:
                dim = len(vect)

    vectors = bcolz.carray(vectors[1:].reshape((idx, dim)),
                           rootdir=str(store_dir / f'{path_to_file.stem}.dat'),
                           mode='w')
    vectors.flush()
    pickle.dump(words, (store_dir / f'{path_to_file.stem}.words.pkl').open('wb'))
    pickle.dump(word2idx, (store_dir / f'{path_to_file.stem}.index.pkl').open('wb'))


def load_glove_vectors(path_to_vectors: PathLike,
                       name: str = '6B',
                       dim: int = 100):
    path_to_vectors = Path(path_to_vectors)
    vectors_name = f'glove.{name}.{dim}d'
    vectors_dir = path_to_vectors / vectors_name
    if not vectors_dir.exists():
        path_raw_glove_vectors = path_to_vectors / f'{vectors_name}.txt'
        if not path_raw_glove_vectors.exists():
            raise ValueError(f"Can't find GloVe word vectors {vectors_name} at {path_to_vectors}")
        preprocess_glove_vectors(path_raw_glove_vectors)
    vectors = bcolz.open(vectors_dir / f'{vectors_name}.dat')[:]
    words = pickle.load((vectors_dir / f'{vectors_name}.words.pkl').open('rb'))
    word2idx = pickle.load((vectors_dir / f'{vectors_name}.index.pkl').open('rb'))
    glove = {w: vectors[word2idx[w]] for w in words}
    return glove


class Embedder(nn.Module):
    def __init__(self,
                 vocab: Vocab,
                 config):
        super(Embedder, self).__init__()
        self.num_embeddings = len(vocab)
        self.embedding_dim = config.embedding_dim
        self.embeddings = nn.Embedding(
            num_embeddings=self.num_embeddings,
            embedding_dim=self.embedding_dim
        )

    def get_size(self):
        return self.embedding_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.embeddings(seq)


class GloveEmbedder(nn.Module):
    def __init__(self,
                 vocab,
                 config):
        super(GloveEmbedder, self).__init__()
        glove = load_glove_vectors(config.path_to_vectors, config.name, config.dim)
        self.embedding_dim = config.dim
        self.num_embeddings = len(vocab)
        weights_matrix = np.zeros((self.num_embeddings, self.embedding_dim), dtype=np.float32)
        for i, token in enumerate(vocab):
            if token in glove:
                weights_matrix[i] = glove[token]
            else:
                weights_matrix[i] = np.random.normal(
                    loc=0.0,
                    scale=GLOVE_STD,
                    size=(self.embedding_dim,)
                )
        self.embeddings = nn.Embedding.from_pretrained(torch.tensor(weights_matrix),
                                                       padding_idx=0,
                                                       freeze=config.get('freeze', False))

    def get_size(self):
        return self.embedding_dim

    def forward(self, seq: torch.Tensor) -> torch.Tensor:
        return self.embeddings(seq)
