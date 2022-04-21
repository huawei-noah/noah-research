# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE

import os
import logging
from tqdm import tqdm, trange
import numpy as np
import torch
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim.adam import Adam
from torch.nn import MSELoss, BCEWithLogitsLoss, CrossEntropyLoss
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler
from xlm_ra import get_slot_labels
from utils import MODEL_CLASSES, compute_metrics, Tasks, set_aux_losses, set_weighting_method

logger = logging.getLogger(__name__)


class Trainer(object):
    def __init__(self, args, train_dataset, dev_dataset, test_dataset,
                 train_examples, dev_examples, test_examples, tokenizer, alignment_dataset):
        self.args = args
        self.alignment_dataset = alignment_dataset
        self.train_dataset = train_dataset
        self.train_examples = train_examples
        self.dev_dataset = dev_dataset
        self.dev_examples = dev_examples
        self.test_dataset = test_dataset
        self.test_examples = test_examples
        self.tokenizer = tokenizer
        self.pad_token_id = args.ignore_index
        self.encoder_class, self.model_class = MODEL_CLASSES[args.model_type]
        self.device = args.cuda_device if torch.cuda.is_available() else "cpu"
        self.model = None  # assigned in the load_model() function

    def train(self):
        train_sampler = RandomSampler(self.train_dataset)
        train_dataloader = DataLoader(self.train_dataset, sampler=train_sampler, batch_size=self.args.train_batch_size)

        t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs
        optimizer = Adam(self.model.parameters(), lr=self.args.learning_rate)
        scheduler = OneCycleLR(optimizer, max_lr=self.args.learning_rate, total_steps=t_total)

        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)

        global_step = 0
        main_loss_float, align_loss = 0.0, 0.0
        self.model.zero_grad()

        # Adaptive Loss Weighting Parameters
        num_aux_losses, use_losses = set_aux_losses(self.args)  # Number of auxiliary losses
        use_weighting = set_weighting_method(self.args)
        loss_steps = 0

        # CoV Weighting parameters
        avg_losses = np.zeros(num_aux_losses)
        avg_loss_ratios = np.ones(num_aux_losses)
        var_loss_ratios = np.zeros(num_aux_losses)
        loss_ratios = np.ones(num_aux_losses)
        cov_losses = np.ones(num_aux_losses)
        epsilon = 1e-8

        loss_weights = np.ones(num_aux_losses)
        avg_loss_weights = np.zeros(num_aux_losses)

        # Logger Parameters
        aux_losses = None

        # Loss configuration logging
        if self.args.align_languages:
            logger.info(f"Num_aux_losses: {num_aux_losses}")
            logger.info(f"Use_losses: {use_losses}")
            logger.info(f"Use_weighting: {use_weighting}")

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration", disable=True)
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch_indices = batch[3].numpy()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                if self.args.task in [Tasks.MTOD.value, Tasks.MTOP.value, Tasks.M_ATIS.value]:
                    inputs = {'input_ids': batch[0], 'intent_labels': batch[1], 'slot_labels': batch[2]}
                else:
                    raise Exception("The task name '%s' is not recognised/supported." % self.args.task)

                english_outputs = self.model(**inputs)
                main_loss = english_outputs[0]
                main_loss_float += main_loss.item()

                if self.args.align_languages:
                    encoder = self.model.roberta
                    batch_target = torch.stack([self.alignment_dataset[index][0] for index in batch_indices]).to(self.device)
                    target_outputs = encoder(input_ids=batch_target)
                    target_cls_embedding = target_outputs[1]
                    target_token_embeddings = target_outputs[0]
                    en_cls_embedding = english_outputs[3]
                    en_token_embeddings = english_outputs[2]

                    unweighted_losses = []  # Collect auxiliary losses

                    language_pairs = [
                        (en_cls_embedding, target_cls_embedding),
                    ]

                    token_embeddings = [
                        en_token_embeddings,
                        target_token_embeddings,
                    ]

                    # ---- Contrastive Loss ----
                    if use_losses.use_contrastive:
                        # Contrastive (CLS) representation
                        src_outputs = en_cls_embedding
                        tar_outputs = target_cls_embedding

                        # Contrastive (Mean-pooled Tokens)
                        # src_outputs = torch.mean(en_token_embeddings, dim=1)
                        # tar_outputs = torch.mean(target_token_embeddings, dim=1)

                        src_n, tar_n = src_outputs.norm(dim=1)[:, None], tar_outputs.norm(dim=1)[:, None]
                        a_norm = src_outputs / src_n
                        b_norm = tar_outputs / tar_n
                        sim = torch.mm(a_norm, b_norm.transpose(0, 1)).to(self.device)

                        labels = torch.arange(sim.shape[0]).to(self.device)

                        loss_fn = CrossEntropyLoss()
                        contrastive_loss = loss_fn(input=sim, target=labels)
                        unweighted_losses.append(contrastive_loss)
                    # ---- Contrastive Loss ----

                    # ---- XeroAlign ----
                    if use_losses.use_xeroalign:
                        loss_fn = MSELoss()
                        xero_align_loss = None
                        for (anchor, target) in language_pairs:
                            if xero_align_loss is None:
                                xero_align_loss = loss_fn(input=anchor, target=target)
                            else:
                                xero_align_loss += loss_fn(input=anchor, target=target)

                        xero_align_loss = xero_align_loss / len(language_pairs)
                        unweighted_losses.append(xero_align_loss)
                    # ---- XeroAlign ----

                    # ---- CrossAligner ----
                    if use_losses.use_crossaligner:
                        slot_labels = torch.stack([self.alignment_dataset[index][2] for index in batch_indices]).to(self.device)
                        loss_fn = BCEWithLogitsLoss()

                        cross_aligner_loss = None
                        for token_embedding in token_embeddings:
                            slot_logits = self.model.slot_classifier(token_embedding)
                            slot_logits = self.model.cross_aligner(slot_logits)

                            if cross_aligner_loss is None:
                                cross_aligner_loss = loss_fn(input=slot_logits, target=slot_labels)
                            else:
                                cross_aligner_loss += loss_fn(input=slot_logits, target=slot_labels)

                        unweighted_losses.append(cross_aligner_loss)
                    # ---- CrossAligner ----

                    # ---- Translate Intent ----
                    if use_losses.use_translate_intent:
                        loss_fct = CrossEntropyLoss()
                        intent_logits = self.model.intent_classifier(target_cls_embedding)
                        intent_labels = torch.stack([self.alignment_dataset[index][1] for index in batch_indices]).to(self.device)
                        intent_loss = loss_fct(input=intent_logits, target=intent_labels)
                        unweighted_losses.append(intent_loss)
                    # ---- Translate Intent ----

                    # Calculate Cov weights
                    prev_loss_step = loss_steps
                    loss_steps += 1

                    aux_losses = np.array([aux_loss.item() for aux_loss in unweighted_losses])
                    # ---- Cov Weighting ----
                    if use_weighting.use_cov:
                        if loss_steps > 1:
                            loss_ratios = aux_losses / (avg_losses + epsilon)
                            # Calculate loss weights
                            cov_losses = np.sqrt(var_loss_ratios + epsilon) / avg_loss_ratios
                            # loss_weights = cov_losses / np.sum(cov_losses)  # loss weight normalisation
                            loss_weights = cov_losses

                        # Update statistics
                        avg_losses = (prev_loss_step * avg_losses + aux_losses) / loss_steps
                        prev_avg_loss_ratios = avg_loss_ratios
                        avg_loss_ratios = (prev_loss_step * avg_loss_ratios + loss_ratios) / loss_steps
                        var_loss_ratios = (prev_loss_step * var_loss_ratios + (loss_ratios - avg_loss_ratios) * (loss_ratios - prev_avg_loss_ratios)) / loss_steps
                    # ---- Cov Weighting ----

                    # Apply weighted sum of aux losses
                    weighted_sum = loss_weights @ aux_losses
                    align_loss += weighted_sum

                    # ---- Update Loss Weight Display Statistic ----
                    avg_loss_weights = (prev_loss_step * avg_loss_weights + loss_weights) / loss_steps
                    # ---- Update Loss Weight Display Statistic ----

                    # Added weighted sum of aux losses to main loss
                    for i in range(num_aux_losses):
                        main_loss += (unweighted_losses[i] * loss_weights[i])

                if self.args.gradient_accumulation_steps > 1:
                    main_loss = main_loss / self.args.gradient_accumulation_steps

                main_loss.backward()  # Only do this once for all losses

                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
                    optimizer.step()
                    scheduler.step()
                    self.model.zero_grad()
                    global_step += 1

            if self.args.task in [Tasks.MTOD.value, Tasks.MTOP.value, Tasks.M_ATIS.value]:
                self.evaluate("dev", exp_name=self.args.model_dir)
            else:
                raise Exception("The task name '%s' is not recognised/supported." % self.args.task)
            logging.info("--------------------------------------")
            logging.info("Train loss after %d steps: %.3f" % (global_step, (main_loss_float / global_step)))
            if self.args.align_languages:
                logging.info("Align Loss after %d steps: %.3f" % (global_step, align_loss / max(1, global_step)))
            logging.info("--------------------------------------")
            logging.info("=======================================")
            logging.info(f"Loss_steps: {loss_steps}")
            logger.info(f"Aux_losses: {aux_losses}")
            if use_weighting.use_cov:
                logger.info(f"Loss_ratios: {loss_ratios}")
                logger.info(f"Avg_loss_ratios: {avg_loss_ratios}")
                logger.info(f"Var_loss_ratios: {var_loss_ratios}")
                logger.info(f"Cov_losses: {cov_losses}")
            logger.info(f"Loss_weights: {loss_weights}")
            logger.info(f"Avg_loss_weights: {avg_loss_weights}")
            logging.info("=======================================")

    def evaluate(self, mode, exp_name):
        if mode == 'test':
            dataset = self.test_dataset
            examples = self.test_examples
        elif mode == 'dev':
            dataset = self.dev_dataset
            examples = self.dev_examples
        else:
            raise Exception("Only dev and test dataset available")

        eval_sampler = SequentialSampler(dataset)
        eval_dataloader = DataLoader(dataset, sampler=eval_sampler, batch_size=self.args.eval_batch_size)

        logger.info("***** Running evaluation on %s dataset *****" % mode)
        logger.info("  Num examples = %d" % len(dataset))
        logger.info("  Batch size = %d" % self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        guids = None
        out_intent_labels = None
        out_slot_labels = None
        slot_label_list = get_slot_labels(self.args)
        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating...", disable=True):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0], 'intent_labels': batch[1], 'slot_labels': batch[2]}
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]
                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Collect ids for debugging
            if guids is None:
                guids = batch[3].detach().cpu().numpy()
            else:
                guids = np.append(guids, batch[3].detach().cpu().numpy(), axis=0)

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_labels = inputs['intent_labels'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_labels = np.append(out_intent_labels, inputs['intent_labels'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                slot_preds = slot_logits.detach().cpu().numpy()
                out_slot_labels = inputs["slot_labels"].detach().cpu().numpy()
            else:
                slot_preds = np.append(slot_preds, slot_logits.detach().cpu().numpy(), axis=0)
                out_slot_labels = np.append(out_slot_labels, inputs["slot_labels"].detach().cpu().numpy(), axis=0)

        eval_loss = eval_loss / nb_eval_steps
        results = {"loss": eval_loss}

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        slot_preds = np.argmax(slot_preds, axis=2)
        slot_label_map = {i: label for i, label in enumerate(slot_label_list)}
        out_slot_label_list = [[] for _ in range(out_slot_labels.shape[0])]
        slot_preds_list = [[] for _ in range(out_slot_labels.shape[0])]

        for i in range(out_slot_labels.shape[0]):
            last_label_tag, last_pred_tag = 'O', 'O'
            for j in range(out_slot_labels.shape[1]):
                if out_slot_labels[i, j] != self.pad_token_id:
                    current_label_tag = slot_label_map[out_slot_labels[i][j]]
                    if current_label_tag == 'O':
                        out_slot_label_list[i].append(current_label_tag)
                        last_label_tag = 'O'
                    elif last_label_tag == 'O':
                        out_slot_label_list[i].append("B-" + current_label_tag)
                        last_label_tag = "B-" + current_label_tag
                    elif last_label_tag == "B-" + current_label_tag or last_label_tag == "I-" + current_label_tag:
                        out_slot_label_list[i].append("I-" + current_label_tag)
                        last_label_tag = "I-" + current_label_tag
                    else:
                        out_slot_label_list[i].append("B-" + current_label_tag)
                        last_label_tag = "B-" + current_label_tag

                    current_pred_tag = slot_label_map[slot_preds[i][j]]
                    if current_pred_tag == 'O':
                        slot_preds_list[i].append(current_pred_tag)
                        last_pred_tag = 'O'
                    elif last_pred_tag == 'O':
                        slot_preds_list[i].append("B-" + current_pred_tag)
                        last_pred_tag = "B-" + current_pred_tag
                    elif last_pred_tag == "B-" + current_pred_tag or last_pred_tag == "I-" + current_pred_tag:
                        slot_preds_list[i].append("I-" + current_pred_tag)
                        last_pred_tag = "I-" + current_pred_tag
                    else:
                        slot_preds_list[i].append("B-" + current_pred_tag)
                        last_pred_tag = "B-" + current_pred_tag

        total_result = compute_metrics(intent_preds, out_intent_labels, slot_preds_list,
                                       out_slot_label_list, examples, guids, self.args)
        results.update(total_result)

        logger.info("********* %s results for %s *********" % (mode.capitalize(), exp_name))
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))
        logger.info("********* %s results for %s *********" % (mode.capitalize(), exp_name))

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        self.model.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saved model checkpoint to '%s'" % self.args.model_dir)

    def load_model(self, final_eval):
        # Okay, there a few options for loading models...
        if final_eval:
            load_model_path = self.args.load_eval_model if self.args.load_eval_model else self.args.model_dir
        elif self.args.load_train_model:
            load_model_path = self.args.load_train_model
        else:
            load_model_path = self.args.model_name_or_path
        if not os.path.exists(load_model_path):
            raise Exception("Model path '%s' doesn't exists! Train first, please..." % load_model_path)
        try:
            logger.info("*****************************************************")
            logger.info("***** Loading Model from '%s' *****" % load_model_path)
            self.model = self.model_class.from_pretrained(load_model_path, args=self.args)
            self.model.to(self.device)
            logger.info("*****************************************************")
        except Exception as e:
            raise Exception(e)
