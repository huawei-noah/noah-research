# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import contextlib
import logging
import os
from collections import OrderedDict

import numpy as np
import torch
from fairseq import metrics, options, utils
from fairseq.data import (Dictionary, LanguagePairDataset,
                          MultilingualCorpusSampledDataset,
                          RoundRobinZipDatasets, TransformEosLangPairDataset)
from fairseq.models import BaseActor, FairseqMultiModel
from fairseq.tasks.translation import load_langpair_dataset
from torch.distributions import Categorical

from . import LegacyFairseqTask, register_task
from .multilingual_translation import MultilingualTranslationTask

logger = logging.getLogger(__name__)


def _lang_token(lang: str):
    return "__{}__".format(lang)


def _lang_token_index(dic: Dictionary, lang: str):
    """Return language token index."""
    idx = dic.index(_lang_token(lang))
    assert idx != dic.unk_index, "cannot find language token for lang {}".format(lang)
    return idx


@register_task("multiuat_multilingual_translation")
class MultiUATMultilingualTranslationTask(MultilingualTranslationTask):
    """A task for training multiple translation models simultaneously.

    We iterate round-robin over batches from multiple language pairs, ordered
    according to the `--lang-pairs` argument.

    The training loop is roughly:

        for i in range(len(epoch)):
            for lang_pair in args.lang_pairs:
                batch = next_batch_for_lang_pair(lang_pair)
                loss = criterion(model_for_lang_pair(lang_pair), batch)
                loss.backward()
            optimizer.step()

    In practice, `next_batch_for_lang_pair` is abstracted in a FairseqDataset
    (e.g., `RoundRobinZipDatasets`) and `model_for_lang_pair` is a model that
    implements the `FairseqMultiModel` interface.

    During inference it is required to specify a single `--source-lang` and
    `--target-lang`, which indicates the inference langauge direction.
    `--lang-pairs`, `--encoder-langtok`, `--decoder-langtok` have to be set to
    the same value as training.
    """

    @staticmethod
    def add_args(parser):
        """Add task-specific arguments to the parser."""
        # fmt: off
        parser.add_argument('data', metavar='DIR', help='path to data directory')
        parser.add_argument('--lang-pairs', default=None, metavar='PAIRS',
                            help='comma-separated list of language pairs (in training order): en-de,en-fr,de-fr')
        parser.add_argument('-s', '--source-lang', default=None, metavar='SRC',
                            help='source language (only needed for inference)')
        parser.add_argument('-t', '--target-lang', default=None, metavar='TARGET',
                            help='target language (only needed for inference)')
        parser.add_argument('--left-pad-source', default='True', type=str, metavar='BOOL',
                            help='pad the source on the left (default: True)')
        parser.add_argument('--left-pad-target', default='False', type=str, metavar='BOOL',
                            help='pad the target on the left (default: False)')
        parser.add_argument('--max-source-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the source sequence')
        parser.add_argument('--max-target-positions', default=1024, type=int, metavar='N',
                            help='max number of tokens in the target sequence')
        parser.add_argument('--upsample-primary', default=1, type=int,
                            help='amount to upsample primary dataset')
        parser.add_argument('--encoder-langtok', default=None, type=str, choices=['src', 'tgt'],
                            metavar='SRCTGT',
                            help='replace beginning-of-sentence in source sentence with source or target '
                                 'language token. (src/tgt)')
        parser.add_argument('--decoder-langtok', action='store_true',
                            help='replace beginning-of-sentence in target sentence with target language token')
        # fmt: on
        parser.add_argument("--temperature", default=1, type=float, help="temperature")
        parser.add_argument("--max-sample-tokens", default=1024, type=float)
        parser.add_argument("--data-actor", default="base", type=str)
        parser.add_argument("--data-actor-optim-step", default=200, type=int)
        parser.add_argument("--data-actor-lr", default=0.0001, type=float)
        parser.add_argument("--update-sampling-interval", default=1000, type=float)
        parser.add_argument("--sample-prob-log", default="", type=str)
        parser.add_argument("--reward-type", default="entropy", type=str)
        parser.add_argument("--K", default=1, type=int)



    def __init__(self, args, dicts, training):
        super().__init__(args, dicts, training)
        self.args = args
        self.dicts = dicts
        self.training = training
        if training:
            self.lang_pairs = args.lang_pairs
        else:
            self.lang_pairs = ["{}-{}".format(args.source_lang, args.target_lang)]
        # eval_lang_pairs for multilingual translation is usually all of the
        # lang_pairs. However for other multitask settings or when we want to
        # optimize for certain languages we want to use a different subset. Thus
        # the eval_lang_pairs class variable is provided for classes that extend
        # this class.
        self.eval_lang_pairs = self.lang_pairs
        # model_lang_pairs will be used to build encoder-decoder model pairs in
        # models.build_model(). This allows multitask type of sub-class can
        # build models other than the input lang_pairs
        self.model_lang_pairs = self.lang_pairs
        self.langs = list(dicts.keys())

        self.data_actor_pretrained = False
        if self.args.sample_prob_log is not None and os.path.exists(self.args.sample_prob_log):
            os.remove(self.args.sample_prob_log)
        self.current_sampling_update_num = 0

    @classmethod
    def setup_task(cls, args, **kwargs):
        dicts, training = cls.prepare(args, **kwargs)
        return cls(args, dicts, training)

    @classmethod
    def update_args(cls, args):
        args.left_pad_source = utils.eval_bool(args.left_pad_source)
        args.left_pad_target = utils.eval_bool(args.left_pad_target)

        if args.lang_pairs is None:
            raise ValueError(
                "--lang-pairs is required. List all the language pairs in the training objective."
            )
        if isinstance(args.lang_pairs, str):
            args.lang_pairs = args.lang_pairs.split(",")

    @classmethod
    def prepare(cls, args, **kargs):
        cls.update_args(args)
        sorted_langs = sorted(
            list({x for lang_pair in args.lang_pairs for x in lang_pair.split("-")})
        )
        if args.source_lang is not None or args.target_lang is not None:
            training = False
        else:
            training = True

        # load dictionaries
        dicts = OrderedDict()
        for lang in sorted_langs:
            paths = utils.split_paths(args.data)
            assert len(paths) > 0
            dicts[lang] = cls.load_dictionary(
                os.path.join(paths[0], "dict.{}.txt".format(lang))
            )
            if len(dicts) > 0:
                assert dicts[lang].pad() == dicts[sorted_langs[0]].pad()
                assert dicts[lang].eos() == dicts[sorted_langs[0]].eos()
                assert dicts[lang].unk() == dicts[sorted_langs[0]].unk()
            if args.encoder_langtok is not None or args.decoder_langtok:
                for lang_to_add in sorted_langs:
                    dicts[lang].add_symbol(_lang_token(lang_to_add))
            logger.info("[{}] dictionary: {} types".format(lang, len(dicts[lang])))
        return dicts, training

    def get_encoder_langtok(self, src_lang, tgt_lang):
        if self.args.encoder_langtok is None:
            return self.dicts[src_lang].eos()
        if self.args.encoder_langtok == "src":
            return _lang_token_index(self.dicts[src_lang], src_lang)
        else:
            return _lang_token_index(self.dicts[src_lang], tgt_lang)

    def get_decoder_langtok(self, tgt_lang):
        if not self.args.decoder_langtok:
            return self.dicts[tgt_lang].eos()
        return _lang_token_index(self.dicts[tgt_lang], tgt_lang)

    def alter_dataset_langtok(
        self,
        lang_pair_dataset,
        src_eos=None,
        src_lang=None,
        tgt_eos=None,
        tgt_lang=None,
    ):
        if self.args.encoder_langtok is None and not self.args.decoder_langtok:
            return lang_pair_dataset

        new_src_eos = None
        if (
            self.args.encoder_langtok is not None
            and src_eos is not None
            and src_lang is not None
            and tgt_lang is not None
        ):
            new_src_eos = self.get_encoder_langtok(src_lang, tgt_lang)
        else:
            src_eos = None

        new_tgt_bos = None
        if self.args.decoder_langtok and tgt_eos is not None and tgt_lang is not None:
            new_tgt_bos = self.get_decoder_langtok(tgt_lang)
        else:
            tgt_eos = None

        return TransformEosLangPairDataset(
            lang_pair_dataset,
            src_eos=src_eos,
            new_src_eos=new_src_eos,
            tgt_bos=tgt_eos,
            new_tgt_bos=new_tgt_bos,
        )

    def load_dataset(self, split, epoch=1, **kwargs):
        """Load a dataset split."""
        paths = utils.split_paths(self.args.data)
        assert len(paths) > 0
        data_path = paths[(epoch - 1) % len(paths)]

        def language_pair_dataset(lang_pair):
            src, tgt = lang_pair.split("-")
            langpair_dataset = load_langpair_dataset(
                data_path,
                split,
                src,
                self.dicts[src],
                tgt,
                self.dicts[tgt],
                combine=True,
                dataset_impl=self.args.dataset_impl,
                upsample_primary=self.args.upsample_primary,
                left_pad_source=self.args.left_pad_source,
                left_pad_target=self.args.left_pad_target,
                max_source_positions=self.args.max_source_positions,
                max_target_positions=self.args.max_target_positions,
            )
            return self.alter_dataset_langtok(
                langpair_dataset,
                src_eos=self.dicts[src].eos(),
                src_lang=src,
                tgt_eos=self.dicts[tgt].eos(),
                tgt_lang=tgt,
            )

        self.datasets[split] = MultilingualCorpusSampledDataset(
            self.args,
            OrderedDict(
                [
                    (lang_pair, language_pair_dataset(lang_pair))
                    for lang_pair in self.lang_pairs
                ]
            ),
            eval_key=None
            if self.training
            else "%s-%s" % (self.args.source_lang, self.args.target_lang),
        )
        if split == "train":
            self.datasets[split].get_sample_prob()
            self.write_sampling_log(self.lang_pairs)
            self.write_sampling_log(self.datasets["train"].p.tolist())


    def build_dataset_for_inference(self, src_tokens, src_lengths, constraints=None):
        if constraints is not None:
            raise NotImplementedError(
                "Constrained decoding with the multilingual_translation task is not supported"
            )

        lang_pair = "%s-%s" % (self.args.source_lang, self.args.target_lang)
        return MultilingualCorpusSampledDataset(
            self.args, 
            OrderedDict(
                [
                    (
                        lang_pair,
                        self.alter_dataset_langtok(
                            LanguagePairDataset(
                                src_tokens, src_lengths, self.source_dictionary
                            ),
                            src_eos=self.source_dictionary.eos(),
                            src_lang=self.args.source_lang,
                            tgt_eos=self.target_dictionary.eos(),
                            tgt_lang=self.args.target_lang,
                        ),
                    )
                ]
            ),
            eval_key=lang_pair,
        )

    def build_model(self, args):
        def check_args():
            messages = []
            if (
                len(set(self.args.lang_pairs).symmetric_difference(args.lang_pairs))
                != 0
            ):
                messages.append(
                    "--lang-pairs should include all the language pairs {}.".format(
                        args.lang_pairs
                    )
                )
            if self.args.encoder_langtok != args.encoder_langtok:
                messages.append(
                    "--encoder-langtok should be {}.".format(args.encoder_langtok)
                )
            if self.args.decoder_langtok != args.decoder_langtok:
                messages.append(
                    "--decoder-langtok should {} be set.".format(
                        "" if args.decoder_langtok else "not"
                    )
                )

            if len(messages) > 0:
                raise ValueError(" ".join(messages))

        # Update args -> the fact that the constructor here
        # changes the args object doesn't mean you get the same one here
        self.update_args(args)

        # Check if task args are consistant with model args
        check_args()

        from fairseq import models

        model = models.build_model(args, self)
        if not isinstance(model, FairseqMultiModel):
            raise ValueError(
                "MultilingualTranslationTask requires a FairseqMultiModel architecture"
            )
        return model

    def build_data_actor(self, args):
        if args.data_actor == "base":
            actor = BaseActor(args, len(self.lang_pairs))
        else:
            actor = None
        return actor

    def train_step(
        self, sample, model, criterion, optimizer, update_num, 
        data_actor=None, data_optimizer=None, 
        ignore_grad=False, prepare_fn=None
    ):
        """
        Do forward and backward, and return the loss as computed by *criterion*
        for the given *model* and *sample*.

        Args:
            sample (dict): the mini-batch. The format is defined by the
                :class:`~fairseq.data.FairseqDataset`.
            model (~fairseq.models.BaseFairseqModel): the model
            criterion (~fairseq.criterions.FairseqCriterion): the criterion
            optimizer (~fairseq.optim.FairseqOptimizer): the optimizer
            update_num (int): the current update
            ignore_grad (bool): multiply loss by 0 if this is set to True

        Returns:
            tuple:
                - the loss
                - the sample size, which is used as the denominator for the
                  gradient
                - logging outputs to display while training
        """
        if not self.data_actor_pretrained and data_actor is not None:
            self.pretrain_data_actor(data_actor, data_optimizer)
            self.data_actor_pretrained = True

        if update_num % self.args.update_sampling_interval == 0 and update_num != 0 and data_actor is not None and self.current_sampling_update_num != update_num:
            # train_grad_state = self.copy_grad(model)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

            p = self.update_sample_probability(model, criterion, data_actor, data_optimizer, prepare_fn)
            self.datasets["train"].update_sampling_p(p)
            self.write_sampling_log(self.datasets["train"].p.tolist())
            # self.restore_grad(train_grad_state, model)
            self.current_sampling_update_num = update_num

        model.train()
        model.set_num_updates(update_num)
        lang_pair = sample["tag"]
        with torch.autograd.profiler.record_function("forward"):
            loss, sample_size, logging_output = criterion(model.models[lang_pair], sample)
        if ignore_grad:
            loss *= 0
        with torch.autograd.profiler.record_function("backward"):
            optimizer.backward(loss)

        return loss, sample_size, logging_output

    def update_sample_probability(
        self, model, criterion, data_actor, data_optimizer, prepare_fn
    ):
        
        logger.info("******* Update Sampling Probability *******")

        all_reward_list = []
        for i, valid_key in enumerate(self.datasets["train"].datasets.keys()):
            sample = self.datasets["valid"].get_sample_with_key(valid_key)[valid_key]
            sample, _ = prepare_fn(sample)
            if self.args.reward_type == "enttp":
                r = self.compute_enttp_monta_carlo(model, sample)
            elif self.args.reward_type == "enteos":
                r = self.compute_enteos_monta_carlo(model, sample)
            elif self.args.reward_type == "pretp":
                r = self.compute_pretp_monte_carlo(model, sample)
            elif self.args.reward_type == "exptp":
                r = self.compute_exptp_monte_carlo(model, sample)
            elif self.args.reward_type == "vartp":
                r = self.compute_vartp_monte_carlo(model, sample)
            elif self.args.reward_type == "comtp":
                r = self.compute_comtp_monte_carlo(model, sample)
            elif self.args.reward_type == "xentropy":
                r = self.compute_xentropy(model, criterion, sample)
                r = r / sample["ntokens"]
            else:
                raise RuntimeError("undefined reward")
            all_reward_list.append(r)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        # sim_list = np.mean(np.array(all_ent_list), axis=0).tolist()
        sim_list = all_reward_list
        logger.info("Rewards List: " + "\t".join([str(i) for i in sim_list]))
        feature = torch.ones(1, len(self.datasets["train"].datasets.keys()))
        grad_scale = torch.FloatTensor(sim_list).view(1, -1)
        if torch.cuda.is_available():
            feature = feature.cuda()
            grad_scale = grad_scale.cuda()
        
        for _ in range(self.args.data_actor_optim_step):
            data_actor.zero_grad()
            data_optimizer.zero_grad()
            a_logits = data_actor(feature)
            loss = -torch.nn.functional.log_softmax(a_logits, dim=-1)
            loss = (loss * grad_scale).sum()
            loss.backward()
            data_optimizer.step()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        with torch.no_grad():
            a_logits = data_actor(feature)
            prob = torch.nn.functional.softmax(a_logits, dim=-1)
            
        return prob.data.view(-1).cpu().numpy()

    def compute_xentropy(self, model, criterion, sample):
        model.eval()
        lang_pair = sample["tag"]
        loss, sample_size, logging_output = criterion(model.models[lang_pair], sample)
        model.train()
        return loss.item()

    
    def compute_pretp_monte_carlo(self, model, sample):
        lang_pair = sample["tag"]
        srclang, tgtlang = lang_pair.split("-")
        target_mask = (sample["target"] != self.dicts[tgtlang].pad()).float()
        lst = []
        for i in range(self.args.K):
            net_output = model.models[lang_pair](**sample["net_input"])
            prob = model.models[lang_pair].get_normalized_probs(net_output, log_probs=True)
            prob, _ = torch.max(prob, dim=-1)
            mean_tp = torch.mean(torch.exp(torch.sum(prob * target_mask, dim=-1))).item()
            lst.append(mean_tp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        return 1-np.mean(np.array(lst))
    
    def compute_exptp_monte_carlo(self, model, sample):
        lang_pair = sample["tag"]
        srclang, tgtlang = lang_pair.split("-")
        target_mask = (sample["target"] != self.dicts[tgtlang].pad()).float()
        lst = []
        for i in range(self.args.K):
            net_output = model.models[lang_pair](**sample["net_input"])
            prob = model.models[lang_pair].get_normalized_probs(net_output, log_probs=True)
            prob, _ = torch.max(prob, dim=-1)
            mean_tp = torch.sum(prob*target_mask, dim=-1) / torch.sum(target_mask, dim=-1)
            mean_tp = torch.mean(torch.exp(mean_tp)).item()
            lst.append(mean_tp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()


        return 1-np.mean(np.array(lst))

    def compute_vartp_monte_carlo(self, model, sample):
        lang_pair = sample["tag"]
        srclang, tgtlang = lang_pair.split("-")
        target_mask = (sample["target"] != self.dicts[tgtlang].pad())
        lst = []
        for i in range(self.args.K):
            net_output = model.models[lang_pair](**sample["net_input"])
            prob = model.models[lang_pair].get_normalized_probs(net_output, log_probs=True)
            varlst = []
            for i in range(prob.size(0)):
                p, m = prob[i], target_mask[i]
                p, _ = torch.max(p, dim=-1)
                varlst.append(torch.var(torch.masked_select(p, m)).item())
            mean_tp = sum(varlst) / len(varlst)
            lst.append(mean_tp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.mean(np.array(lst))

    def compute_comtp_monte_carlo(self, model, sample):
        lang_pair = sample["tag"]
        srclang, tgtlang = lang_pair.split("-")
        target_mask = (sample["target"] != self.dicts[tgtlang].pad())
        lst = []
        for i in range(self.args.K):
            net_output = model.models[lang_pair](**sample["net_input"])
            prob = model.models[lang_pair].get_normalized_probs(net_output, log_probs=True)
            varlst = []
            for i in range(prob.size(0)):
                p, m = prob[i], target_mask[i]
                p, _ = torch.max(p, dim=-1)
                varlst.append(torch.var(torch.masked_select(p, m)).item())
            prob, _ = torch.max(prob, dim=-1)
            mean_tp = (torch.sum(prob*target_mask.float(), dim=-1) / torch.sum(target_mask.float(), dim=-1)).detach().cpu().numpy()
            mean_tp = np.mean(np.exp(np.array(varlst) / mean_tp)) 

            lst.append(mean_tp)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.mean(np.array(lst))
    
    def compute_enttp_monta_carlo(self, model, sample):
        lang_pair = sample["tag"]
        srclang, tgtlang = lang_pair.split("-")
        target_mask = (sample["target"] != self.dicts[tgtlang].pad()).float()
        lst = []
        for i  in range(self.args.K):
            net_output = model.models[lang_pair](**sample["net_input"])
            prob = model.models[lang_pair].get_normalized_probs(net_output, log_probs=False)
            e = Categorical(probs=prob).entropy().detach()
            e = (torch.sum(e * target_mask) / torch.sum(target_mask)).cpu().numpy()
            lst.append(e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.mean(np.array(lst))

    def compute_enteos_monta_carlo(self, model, sample):
        lang_pair = sample["tag"]
        srclang, tgtlang = lang_pair.split("-")
        target_mask = (sample["target"] == self.dicts[tgtlang].eos()).float()
        lst = []
        for i  in range(self.args.K):
            net_output = model.models[lang_pair](**sample["net_input"])
            prob = model.models[lang_pair].get_normalized_probs(net_output, log_probs=False)
            e = Categorical(probs=prob).entropy().detach()
            e = (torch.sum(e * target_mask) / torch.sum(target_mask)).cpu().numpy()
            lst.append(e)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        return np.mean(np.array(lst))

    def pretrain_data_actor(self, data_actor, data_optimizer):
        logger.info("******* Pretrain Data Actor *******")
        feature =  torch.ones(1, len(self.datasets["train"].datasets.keys()))
        datasize_p = self.datasets["train"].p
        target = torch.FloatTensor(datasize_p).view(1, -1)
        if torch.cuda.is_available():
            feature = feature.cuda()
            target = target.cuda()
        l = 100
        count = 0
        while l > 0.00000001:
            data_actor.zero_grad()
            data_optimizer.zero_grad()
            a_logits = data_actor(feature)
            prob = torch.nn.functional.softmax(a_logits, dim=-1)
            loss = torch.nn.functional.mse_loss(prob, target)
            l = loss.item()
            if count % 1000 == 0 :
                logger.info("Pretrain Data Actor | Loss = %.7f | num_updates = %10d" % (l, count))
            loss.backward()
            data_optimizer.step()
            # grad = torch.autograd.grad(loss, filter(lambda x:x.requires_grad, data_actor.parameters()))
            # updated_weights = {n: p - self.args.data_actor_lr * 10 * g for g, (n, p) in zip(grad, data_actor.named_parameters())}
            # data_actor = self.update_params(data_actor, updated_weights)
            count += 1

        with torch.no_grad():
            a_logits = data_actor(feature)
            prob = torch.nn.functional.softmax(a_logits, dim=-1)
            sim_list = [i for i in prob.data.view(-1).cpu().numpy()]
            logger.info("******* Pretrain Complete *******")
            for x, y, z in list(zip(self.lang_pairs, self.datasets["train"].dataset_sizes, sim_list)):
                logger.info("Pretrained Data Actor | Domain: %7s | size = %7d | Sampling probability = %6.3f%% " % (x, y, z*100))

    def update_params(self, model, updated_weights):
        for n, p in model.named_parameters():
            p.data = updated_weights[n]
        return model

    def copy_grad(self, model, to_cpu=False):
        state = {}
        for n, p in model.named_parameters():
            state[n] = p.grad.data.clone()
        return state
    
    def restore_grad(self, state, model):
        for n, p in model.named_parameters():
            p.grad.data = state[n]

    def grad_cosine_sim(self, train_grad, valid_grad):
        cosine_prod, train_cosine_norm, valid_cosine_norm = 0, 0, 0
        for (nt, gt), (nv, gv) in zip(train_grad.items(), valid_grad.items()):
            assert nt == nv
            cosine_prod += (gt.data * gv.data).sum().item()
            train_cosine_norm += gt.data.norm(2) ** 2
            valid_cosine_norm += gv.data.norm(2) ** 2

        cosine_sim = cosine_prod / ((train_cosine_norm*valid_cosine_norm)**0.5 + 1e-10)
        return cosine_sim.item()

    def write_sampling_log(self, lst):
        if self.args.sample_prob_log is not None:
            with open(self.args.sample_prob_log, "a", encoding="utf-8") as f:
                f.write(",".join([str(i) for i in lst]) + "\n")

    def valid_step(self, sample, model, criterion):
        lang_pair = sample["tag"]
        loss, sample_size, logging_output = criterion(model.models[lang_pair], sample)
        return loss, sample_size, logging_output


    def inference_step(
        self, generator, models, sample, prefix_tokens=None, constraints=None
    ):
        with torch.no_grad():
            if self.args.decoder_langtok:
                bos_token = _lang_token_index(
                    self.target_dictionary, self.args.target_lang
                )
            else:
                bos_token = self.target_dictionary.eos()
            return generator.generate(
                models,
                sample,
                prefix_tokens=prefix_tokens,
                constraints=constraints,
                bos_token=bos_token,
            )

    def reduce_metrics(self, logging_outputs, criterion):
        with metrics.aggregate():
            # pass 'sample_size', 'nsentences', 'ntokens' stats to fairseq_task
            super().reduce_metrics(logging_outputs, criterion)
            for k in ["sample_size", "nsentences", "ntokens"]:
                metrics.log_scalar(k, sum(l[k] for l in logging_outputs))

    @property
    def source_dictionary(self):
        if self.training:
            return next(iter(self.dicts.values()))
        else:
            return self.dicts[self.args.source_lang]

    @property
    def target_dictionary(self):
        if self.training:
            return next(iter(self.dicts.values()))
        else:
            return self.dicts[self.args.target_lang]

    def max_positions(self):
        """Return the max sentence length allowed by the task."""
        if len(self.datasets.values()) == 0:
            return {
                "%s-%s"
                % (self.args.source_lang, self.args.target_lang): (
                    self.args.max_source_positions,
                    self.args.max_target_positions,
                )
            }
        return OrderedDict(
            [
                (key, (self.args.max_source_positions, self.args.max_target_positions))
                for split in self.datasets.keys()
                for key in self.datasets[split].datasets.keys()
            ]
        )
