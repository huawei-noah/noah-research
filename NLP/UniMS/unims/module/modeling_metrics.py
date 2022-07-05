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
from rouge import Rouge
from torchmetrics import Metric
import torch
import nltk
import clip

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(lineno)d - %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class Accuracy(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("correct", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, preds, target, acc_len=0):
        assert len(preds) == len(target)
        correct = 0.0

        if acc_len > 0:
            preds = preds[:acc_len]
            target = target[:acc_len]
            for pred in preds:
                if pred in target:
                    correct += 1
            self.correct += correct / len(target)
        else:
            correct += sum([1 for i in range(len(preds)) if preds[i] == target[i]])
            self.correct += correct / len(target)

        self.total += 1

    def compute(self):
        return self.correct.float() / self.total


class ROUGEScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        # self.add_state("preds", default=[], dist_reduce_fx="cat")
        # self.add_state("refers", default=[], dist_reduce_fx="cat")
        self.add_state("r1_p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r1_r", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r1_f", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("r2_p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r2_r", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r2_f", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("rl_p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rl_r", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rl_f", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        self.rouge = Rouge()

    def update(self, preds, refers):
        assert len(preds) == len(refers)
        rouge_results = self.rouge.get_scores(preds, refers)
        for rouge_result in rouge_results:
            self.r1_p += rouge_result['rouge-1']['p']
            self.r1_r += rouge_result['rouge-1']['r']
            self.r1_f += rouge_result['rouge-1']['f']

            self.r2_p += rouge_result['rouge-2']['p']
            self.r2_r += rouge_result['rouge-2']['r']
            self.r2_f += rouge_result['rouge-2']['f']

            self.rl_p += rouge_result['rouge-l']['p']
            self.rl_r += rouge_result['rouge-l']['r']
            self.rl_f += rouge_result['rouge-l']['f']

        self.total += len(preds)
        return rouge_results

    def compute(self):
        # return rouge_results
        return {
            'total': self.total.item(),
            'rouge-1': {
                'f': (self.r1_f / self.total).item(),
                'p': (self.r1_p / self.total).item(),
                'r': (self.r1_r / self.total).item(),
            },
            'rouge-2': {
                'f': (self.r2_f / self.total).item(),
                'p': (self.r2_p / self.total).item(),
                'r': (self.r2_r / self.total).item(),
            },
            'rouge-L': {
                'f': (self.rl_f / self.total).item(),
                'p': (self.rl_p / self.total).item(),
                'r': (self.rl_r / self.total).item(),
            }
        }


class BERTScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        from bert_score import BERTScorer

        self.add_state("p", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("r", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("f", default=torch.tensor(0.0), dist_reduce_fx="sum")

        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.bert_scorer = BERTScorer(lang="en", rescale_with_baseline=True)
        self.bert_scorer = BERTScorer(
            model_type="/home/ma-user/work/zhangzhengkun/Dataset/PTM/transformers/roberta-large",
            # lang="en",
            num_layers=17,
            # rescale_with_baseline=True,
            device='cuda:0'
        )

    def update(self, preds, refers):
        assert len(preds) == len(refers)
        # rouge_results = self.rouge.get_scores(preds, refers)
        bert_score_results = self.bert_scorer.score(preds, refers)
        # logger.info(bert_score_results)

        for bert_score_result in bert_score_results[0]:
            self.p += bert_score_result
        for bert_score_result in bert_score_results[1]:
            self.r += bert_score_result
        for bert_score_result in bert_score_results[2]:
            self.f += bert_score_result

        self.total += len(preds)
        return bert_score_results

    def compute(self):
        return {
            'total': self.total.item(),
            'bert_score': {
                'f': (self.p / self.total).item(),
                'p': (self.r / self.total).item(),
                'r': (self.f / self.total).item(),
            }
        }


class MoverScore(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        from moverscore_v2 import get_idf_dict, word_mover_score

        self.get_idf_dict = get_idf_dict
        self.word_mover_score = word_mover_score

        # self.add_state("preds", default=[], dist_reduce_fx="cat")
        # self.add_state("refers", default=[], dist_reduce_fx="cat")
        self.add_state("mover_scores", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")
        # self.rouge = Rouge()

    def update(self, preds, refers):
        assert len(preds) == len(refers)

        idf_dict_hyp = self.get_idf_dict(preds)
        idf_dict_ref = self.get_idf_dict(refers)

        mover_scores = self.word_mover_score(
            refers, preds, idf_dict_ref, idf_dict_hyp,
            stop_words=[],
            n_gram=1,
            remove_subwords=True,
            batch_size=48
        )
        # logger.info(mover_scores)
        for mover_score in mover_scores:
            self.mover_scores += mover_score

        self.total += len(preds)
        return mover_scores

    def compute(self):
        return {
            'total': self.total.item(),
            'mover_scores': (self.mover_scores / self.total).item(),
        }


class Similarity(Metric):
    def __init__(self, model_path, vocab_path, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        from clip.simple_tokenizer import SimpleTokenizer

        # # load model and options
        # checkpoint = torch.load(model_path)
        # opt = checkpoint['opt']
        # # if data_path is not None:
        # #     opt.data_path = data_path
        #
        # # load vocabulary used by the model
        # # with open(join(opt.vocab_path, '%s_vocab.pkl' % opt.data_name), 'rb') as f:
        # #     self.vocab = pk.load(f)
        # with open(vocab_path, 'rb') as f:
        #     self.vocab = pk.load(f)
        # opt.vocab_size = len(self.vocab)
        #
        # # construct model
        # self.model = VSE(opt)
        # # load model state
        # self.model.load_state_dict(checkpoint['model'])
        #
        # self.transform = transforms.Compose([
        #     transforms.Resize(256, interpolation=transforms.InterpolationMode.BICUBIC),
        #     transforms.CenterCrop(256),
        #     lambda image: image.convert("RGB"),
        #     transforms.ToTensor(),
        #     # transforms.Normalize((0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)),
        #     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        # ])
        self.model, self.transform = clip.load(model_path, jit=False)
        self.mm_tokenizer = SimpleTokenizer(vocab_path)
        for param in self.model.parameters():
            param.requires_grad = False

        self.add_state("similarity", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("total", default=torch.tensor(0), dist_reduce_fx="sum")

    def tokenize(self, texts, context_length=77):
        if isinstance(texts, str):
            texts = [texts]

        sot_token = self.mm_tokenizer.encoder["<|startoftext|>"]
        eot_token = self.mm_tokenizer.encoder["<|endoftext|>"]
        all_tokens = [[sot_token] + self.mm_tokenizer.encode(
            text)[:context_length - 2] + [eot_token] for text in texts]
        result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

        for i, tokens in enumerate(all_tokens):
            if len(tokens) > context_length:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}")
            result[i, :len(tokens)] = torch.tensor(tokens)

        return result

    def preprocess(self, image, caption):
        image = self.transform(image)
        # Convert caption (string) to word ids.
        tokens = nltk.tokenize.word_tokenize(str(caption).lower())
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        target = torch.Tensor(caption)
        return image, target

    def collate_fn(self, data):
        # Sort a data list by caption length
        data.sort(key=lambda x: len(x[1]), reverse=True)
        # images, captions, ids, img_ids = zip(*data)
        images, captions = zip(*data)

        # Merge images (convert tuple of 3D tensor to 4D tensor)
        images = torch.stack(images, 0)

        # Merget captions (convert tuple of 1D tensor to 2D tensor)
        lengths = [len(cap) for cap in captions]
        targets = torch.zeros(len(captions), max(lengths)).long()
        for i, cap in enumerate(captions):
            end = lengths[i]
            targets[i, :end] = cap[:end]

        return images, targets, lengths

    def update(self, images, captions):
        assert len(images) == len(captions)
        # data = [self.preprocess(image, caption) for image, caption in zip(images, captions)]
        # images, targets, lengths = self.collate_fn(data)
        # img_emb, cap_emb = self.model.forward_emb(images, targets, lengths)
        # sim = cosine_sim(img_emb, cap_emb)
        # logger.info(sim)
        # logger.info(torch.diag(sim))
        # logger.info(sim.diag())
        # logger.info(sim.diag().sum())
        # self.similarity += sim.diag().sum()

        images = [self.transform(image) for image in images]
        images = torch.stack(images, dim=0).cuda()
        captions = self.tokenize(captions).cuda()
        logits_per_image, _ = self.model(images, captions)
        logits_per_image = logits_per_image.diag()
        self.similarity += logits_per_image.sum()
        self.total += len(captions)
        return logits_per_image.detach().cpu().numpy()

    def compute(self):
        return self.similarity.float() / self.total.float()
