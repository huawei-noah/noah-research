# Copyright (C) 2022. Huawei Technologies Co., Ltd. All rights reserved.
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
# ============================================================================

import torch
import numpy as np
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

InputDataClass = NewType("InputDataClass", Any)

with open("languages/xlmr_langs.txt", "r") as infile:
    LANG_TAGS_START, LANG_TAGS_END = [], []
    for line in infile:
        line = line.rstrip()
        LANG_TAGS_START.append(f"<{line}>")
        LANG_TAGS_END.append(f"</{line}>")

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """
    Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary.
    """
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.
    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (
        pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0
    ):
        return torch.stack(examples, dim=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(x.size(0) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)

    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example

    return result


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "pt":
            return self.torch_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


@dataclass
class DynamicDataCollatorForLanguageModeling(DataCollatorMixin):
    """
    Data collator used for language modeling. Inputs are dynamically padded to the maximum length of a batch if they
    are not all of the same length.
    tokenization on the fly
    """

    def __init__(
        self,
        tokenizer,
        mlm_probability=0.15,
        pad_to_multiple_of=None,
        return_tensors="pt",
        id2lang=None,
        max_length=None,
    ):
        self.tokenizer = tokenizer
        self.mlm_probability = mlm_probability
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.id2lang = id2lang
        self.lang2id = {v: k for k, v in id2lang.items()}
        self.max_length = max_length
        self.mlm = True

    def torch_call(self, features):

        # ----- Tokenization on the fly ----- #
        examples = {"language": [], "id": []}
        cs_sents = []
        for f in features:
            examples["language"].append(self.lang2id[f["language"]])
            examples["id"].append(f["id"])
            # remove lang tags, for one cs_sentence it can have the <l></l> and <en></en>
            no_tag_cs_sentence = (
                f["cs_sentence"]
                .replace(f"<{f['language']}>", "")
                .replace(f"</{f['language']}>", "")
                .replace("<en>", "")
                .replace("</en>", "")
            )

            cs_sents.append(no_tag_cs_sentence)

        examples.update(
            self.tokenizer(
                cs_sents,
                padding="max_length",
                truncation=True,
                max_length=self.max_length,
                return_special_tokens_mask=True,
            )
        )
        examples["language"] = torch.tensor(examples["language"]).long()
        examples["id"] = torch.tensor(examples["id"]).long()

        batch = self.tokenizer.pad(
            examples, return_tensors="pt", pad_to_multiple_of=self.pad_to_multiple_of
        )

        # If special token mask has been preprocessed, pop it from the dict.
        special_tokens_mask = batch.pop("special_tokens_mask", None)
        if self.mlm:
            batch["input_ids"], batch["labels"] = self.torch_mask_tokens(
                batch["input_ids"], special_tokens_mask=special_tokens_mask
            )
        else:
            labels = batch["input_ids"].clone()
            if self.tokenizer.pad_token_id is not None:
                labels[labels == self.tokenizer.pad_token_id] = -100
            batch["labels"] = labels

        return batch

    def torch_mask_tokens(
        self, inputs: Any, special_tokens_mask: Optional[Any] = None
    ) -> Tuple[Any, Any]:
        """
        Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
        """

        labels = inputs.clone()
        # We sample a few tokens in each sequence for MLM training (with probability `self.mlm_probability`)
        probability_matrix = torch.full(labels.shape, self.mlm_probability)
        if special_tokens_mask is None:
            special_tokens_mask = [
                self.tokenizer.get_special_tokens_mask(
                    val, already_has_special_tokens=True
                )
                for val in labels.tolist()
            ]
            special_tokens_mask = torch.tensor(special_tokens_mask, dtype=torch.bool)
        else:
            special_tokens_mask = special_tokens_mask.bool()

        probability_matrix.masked_fill_(special_tokens_mask, value=0.0)
        masked_indices = torch.bernoulli(probability_matrix).bool()
        labels[~masked_indices] = -100  # We only compute loss on masked tokens

        # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
        indices_replaced = (
            torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
        )
        inputs[indices_replaced] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )

        # 10% of the time, we replace masked input tokens with random word
        indices_random = (
            torch.bernoulli(torch.full(labels.shape, 0.5)).bool()
            & masked_indices
            & ~indices_replaced
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )
        inputs[indices_random] = random_words[indices_random]

        # The rest of the time (10% of the time) we keep the masked input tokens unchanged
        return inputs, labels


@dataclass
class DynamicDataCollatorForEntityMasking(DataCollatorMixin):
    """
    Data collator used for language modeling that masks entire words.
    - collates batches of tensors, honoring their tokenizer's pad_token
    - preprocesses batches for masked language modeling
    """

    def __init__(
        self,
        tokenizer,
        pad_to_multiple_of=None,
        return_tensors="pt",
        id2lang=None,
        entity_probability=None,
        masking=None,
        partial_masking=None,
        keep_random=None,
        keep_same=None,
        insert_lang_tag=None,
        max_length=None,
    ):
        self.tokenizer = tokenizer
        self.pad_to_multiple_of = pad_to_multiple_of
        self.return_tensors = return_tensors
        self.id2lang = id2lang
        self.lang2id = {v: k for k, v in id2lang.items()}
        self.max_length = max_length
        self.masking = masking
        self.insert_lang_tag = insert_lang_tag
        self.partial_masking = partial_masking
        self.entity_probability = entity_probability
        self.keep_random = keep_random
        self.keep_same = keep_same

    def torch_call(self, features):
        # ----- Tokenization on the fly ----- #
        tokens = {"language": [], "id": []}
        cs_sents = []
        for f in features:
            tokens["language"].append(self.lang2id[f["language"]])
            tokens["id"].append(f["id"])
            cs_sents.append(f["cs_sentence"])

        tokenized_batch = self.tokenizer(
            cs_sents,
            truncation=True,
            max_length=self.max_length,
            return_special_tokens_mask=True,
        )
        special_tokens_mask = tokenized_batch.pop("special_tokens_mask", None)
        tokens.update(tokenized_batch)

        tokens["language"] = torch.tensor(tokens["language"]).long()
        tokens["id"] = torch.tensor(tokens["id"]).long()
        # ------------------------------------- #

        # masking_indices_candidates_entities, mask_indices_entities and random_indices_entities are inside entities
        # masking_indices_candidates_entities are the chosen indices for further 80%/10%/10%
        # all ~masking_indices_candidates_entities labels are first marked as -100
        # then 80% of masking_indices_candidates_entities are set to mask_indices_entities, where input is replaced with [MASK]
        # and 10% of masking_indices_candidates_entities are set to random_indices_entities, where input is replaced with random id
        # the rest 10% we do not change input, but the corresponding labels are still input rather than -100
        input_ids, attention_mask, languages = [], [], []
        # if partial masking, we also have random_indices_entities
        (
            masking_indices_candidates_entities,
            mask_indices_entities,
            random_indices_entities,
        ) = ([], [], [])
        # if ep-mlm we also have non_entities masking indices
        (
            masking_indices_candidates_non_entities,
            mask_indices_non_entities,
            random_indices_non_entities,
        ) = ([], [], [])
        langs_not_used = {}

        # Go through all examples
        for j, (inp_ids, att_m, lang, id_, ex_special_tokens_mask) in enumerate(
            zip(
                tokens["input_ids"],
                tokens["attention_mask"],
                tokens["language"],
                tokens["id"],
                special_tokens_mask,
            )
        ):
            ref_tokens = []
            for token_id in inp_ids:
                token = self.tokenizer._convert_id_to_token(token_id)
                ref_tokens.append(token)

            # first get mask labels for tokens inside entities
            (
                ex_masking_indices_candidates_entities,
                ex_mask_indices_entities,
                ex_random_indices_entities,
                ex_lang_tags_indices,
                sentences_not_used,
                cand_indexes,
            ) = self.get_entities_masking(ref_tokens, lang.item())

            for k, v in sentences_not_used.items():
                langs_not_used[k] = langs_not_used.get(k, 0) + 1

            # then get mask labels for tokens outside entities if we further do mlm
            if self.masking == "ep-mlm":
                # now we continue to mask 15% of the tokens outside of the entity tags 80% of the time, 10% unchanged,
                # 10% random, remember to set special_tokens_mask to False
                ex_probability_matrix = torch.full((1, len(ref_tokens)), 0.15).squeeze(0)

                ex_special_tokens_mask = torch.tensor(ex_special_tokens_mask).bool()
                ex_probability_matrix.masked_fill_(ex_special_tokens_mask, value=0.0)

                ex_masking_indices_candidates_non_entities = torch.bernoulli(ex_probability_matrix).bool()

                if len(cand_indexes) > 0:
                    # cand_indexes is [[idx in entity1], [idx in entity2], ...]
                    # also add the language tags ids they wont be masked
                    flat_entity_and_tags_idx = [
                        item for sublist in cand_indexes for item in sublist
                    ] + ex_lang_tags_indices

                    for idx in flat_entity_and_tags_idx:
                        # only get the mask labels outside entity and tags
                        ex_masking_indices_candidates_non_entities[idx] = False

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                ex_mask_indices_non_entities = (
                    torch.bernoulli(torch.full((1, len(ref_tokens)), 0.8)).bool()
                    & ex_masking_indices_candidates_non_entities
                ).squeeze(0)

                # 10% of the time, we replace masked input tokens with random word
                ex_random_indices_non_entities = (
                    torch.bernoulli(torch.full((1, len(ref_tokens)), 0.5)).bool()
                    & ex_masking_indices_candidates_non_entities
                    & ~ex_mask_indices_non_entities
                ).squeeze(0)

            else:
                (
                    ex_masking_indices_candidates_non_entities,
                    ex_mask_indices_non_entities,
                    ex_random_indices_non_entities,
                ) = (
                    torch.tensor([False] * len(ref_tokens)),
                    torch.tensor([False] * len(ref_tokens)),
                    torch.tensor([False] * len(ref_tokens)),
                )

            # Explicitly remove Language tags and add them at the beginning of the sentence
            (
                ex_input_ids,
                ex_attention_mask,
                ex_masking_indices_candidates_entities,
                ex_mask_indices_entities,
                ex_random_indices_entities,
                ex_masking_indices_candidates_non_entities,
                ex_mask_indices_non_entities,
                ex_random_indices_non_entities,
            ) = self.remove_language_tags(
                inp_ids,
                att_m,
                ex_lang_tags_indices,
                lang.item(),
                ex_masking_indices_candidates_entities,
                ex_mask_indices_entities,
                ex_random_indices_entities,
                ex_masking_indices_candidates_non_entities.tolist(),
                ex_mask_indices_non_entities.tolist(),
                ex_random_indices_non_entities.tolist(),
            )

            assert (
                len(ex_input_ids)
                == len(ex_attention_mask)
                == len(ex_mask_indices_entities)
                == len(ex_masking_indices_candidates_entities)
                == len(ex_random_indices_entities)
                == len(ex_mask_indices_non_entities)
                == len(ex_random_indices_non_entities)
                == len(ex_masking_indices_candidates_non_entities)
            )

            input_ids.append(ex_input_ids)
            attention_mask.append(torch.tensor(ex_attention_mask).long())
            languages.append(lang.item())
            masking_indices_candidates_entities.append(
                torch.tensor(ex_masking_indices_candidates_entities).long()
            )
            mask_indices_entities.append(torch.tensor(ex_mask_indices_entities).long())
            random_indices_entities.append(
                torch.tensor(ex_random_indices_entities).long()
            )
            masking_indices_candidates_non_entities.append(
                torch.tensor(ex_masking_indices_candidates_non_entities).long()
            )
            mask_indices_non_entities.append(
                torch.tensor(ex_mask_indices_non_entities).long()
            )
            random_indices_non_entities.append(
                torch.tensor(ex_random_indices_non_entities).long()
            )

        # move collate here - after removing tags for each sentence
        batch_input = _torch_collate_batch(
            input_ids, self.tokenizer, pad_to_multiple_of=self.pad_to_multiple_of
        )
        attention_mask = torch.nn.utils.rnn.pad_sequence(
            attention_mask, batch_first=True, padding_value=0
        )
        batch_masking_indices_candidates_entities = torch.nn.utils.rnn.pad_sequence(
            masking_indices_candidates_entities, batch_first=True, padding_value=0
        )
        batch_mask_indices_entities = torch.nn.utils.rnn.pad_sequence(
            mask_indices_entities, batch_first=True, padding_value=0
        )
        batch_random_indices_entities = torch.nn.utils.rnn.pad_sequence(
            random_indices_entities, batch_first=True, padding_value=0
        )
        batch_masking_indices_candidates_non_entities = torch.nn.utils.rnn.pad_sequence(
            masking_indices_candidates_non_entities, batch_first=True, padding_value=0
        )
        batch_mask_indices_non_entities = torch.nn.utils.rnn.pad_sequence(
            mask_indices_non_entities, batch_first=True, padding_value=0
        )
        batch_random_indices_non_entities = torch.nn.utils.rnn.pad_sequence(
            random_indices_non_entities, batch_first=True, padding_value=0
        )

        inputs, labels = self.torch_mask_tokens(
            batch_input,
            batch_masking_indices_candidates_entities.bool(),
            batch_mask_indices_entities.bool(),
            batch_random_indices_entities.bool(),
            batch_masking_indices_candidates_non_entities.bool(),
            batch_mask_indices_non_entities.bool(),
            batch_random_indices_non_entities.bool(),
        )

        langs_not_used_tensor = {
            k: torch.tensor(v).long() for k, v in langs_not_used.items()
        }

        return {
            "input_ids": inputs,
            "labels": labels,
            "attention_mask": attention_mask,
            "language": torch.tensor(languages).long(),
            "langs_not_used": langs_not_used_tensor,
        }

    def remove_language_tags(
        self,
        inputs,
        attention_mask,
        lang_tags,
        lang,
        entities_masking_candidates,
        entities_mask,
        entities_random=None,
        non_entities_masking_candidates=None,
        non_entities_mask=None,
        non_entities_random=None,
    ):
        """
        Remove Language tags around entities
        """

        new_inputs, new_attention_mask = [], []
        new_entities_masking_candidates, new_entities_mask, new_entities_random = ([], [], [])
        (
            new_non_entities_masking_candidates,
            new_non_entities_mask,
            new_non_entities_random,
        ) = ([], [], [])
        for i, _ in enumerate(inputs):
            if i in lang_tags:  # if idx in lang_tags positions, pass
                continue
            else:
                new_inputs.append(inputs[i])
                new_attention_mask.append(attention_mask[i])
                new_entities_masking_candidates.append(entities_masking_candidates[i])
                new_entities_mask.append(entities_mask[i])
                if entities_random is not None:
                    new_entities_random.append(entities_random[i])
                if (
                    non_entities_mask is not None
                    and non_entities_random is not None
                    and non_entities_masking_candidates is not None
                ):
                    new_non_entities_masking_candidates.append(
                        non_entities_masking_candidates[i]
                    )
                    new_non_entities_mask.append(non_entities_mask[i])
                    new_non_entities_random.append(non_entities_random[i])

        if self.insert_lang_tag:
            # Add a language tag at the beginning of the sentence (after CLS)
            new_inputs.insert(
                1, self.tokenizer.convert_tokens_to_ids(f"<{self.id2lang[lang]}>")
            )
            new_attention_mask.insert(1, 1)
            new_entities_masking_candidates.insert(1, False)
            new_entities_mask.insert(1, False)
            if entities_random is not None:
                new_entities_random.insert(1, False)
            if (
                non_entities_mask is not None
                and non_entities_random is not None
                and non_entities_masking_candidates is not None
            ):
                new_non_entities_masking_candidates.insert(1, False)
                new_non_entities_mask.insert(1, False)
                new_non_entities_random.insert(1, False)
        return (
            new_inputs,
            new_attention_mask,
            new_entities_masking_candidates,
            new_entities_mask,
            new_entities_random,
            new_non_entities_masking_candidates,
            new_non_entities_mask,
            new_non_entities_random,
        )

    def get_entities_masking(self, input_tokens, language):

        sentences_not_used = {}
        cand_indexes = []
        adding_cand_id = False
        lang_tag_indices = []

        tmp_entity = []
        for (i, token) in enumerate(input_tokens):
            # traverse the list to until find a LANG_TAGS_START:
            if token in LANG_TAGS_START:
                # this is the start of language tag, meaning entity will start from the following index
                adding_cand_id = True
                lang_tag_indices += [i]
                tmp_entity = []

            elif token in LANG_TAGS_END and adding_cand_id:
                adding_cand_id = False
                lang_tag_indices += [i]
                if tmp_entity:
                    cand_indexes.append(tmp_entity)
                    tmp_entity = []
                else:
                    sentences_not_used[self.id2lang[language]] = (
                        sentences_not_used.get(self.id2lang[language], 0) + 1
                    )

            else:
                if adding_cand_id:
                    tmp_entity.append(i)

        if len(cand_indexes) > 0:

            flat_entity_idx = [item for sublist in cand_indexes for item in sublist]

            if self.partial_masking:
                # partial entity masking (PEP)
                probability_matrix = torch.full(
                    (1, len(input_tokens)), self.entity_probability
                ).squeeze(0)

                for idx in range(len(input_tokens)):
                    if not idx in flat_entity_idx:
                        # only get the probs inside entity
                        probability_matrix[idx] = 0.0

                # masking_indices_candidates will further be marked as indices_mask [MASK], indices_random and rest
                masking_indices_candidates = torch.bernoulli(probability_matrix).bool()

                # 80% of the time, we replace input tokens with tokenizer.mask_token ([MASK])
                bernoulli = torch.bernoulli(
                    torch.full((1, len(input_tokens)), 0.8)
                ).bool()

                indices_mask = (bernoulli & masking_indices_candidates).squeeze(0)

                if self.keep_random:
                    # 10% of the time, we replace masked input tokens with random word
                    indices_random = (
                        torch.bernoulli(torch.full((1, len(input_tokens)), 0.5)).bool()
                        & masking_indices_candidates
                        & ~indices_mask
                    ).squeeze(0)

                    return (
                        masking_indices_candidates.tolist(),
                        indices_mask.tolist(),
                        indices_random.tolist(),
                        lang_tag_indices,
                        sentences_not_used,
                        cand_indexes,
                    )

                return (
                    masking_indices_candidates.tolist(),
                    indices_mask.tolist(),
                    [False] * len(input_tokens),
                    lang_tag_indices,
                    sentences_not_used,
                    cand_indexes,
                )
            else:
                # whole entity masking (WEP)

                # entity_idx is a dictionary {start_entity_idx: [the rest of the entity idx in the same entity tags}
                entity_idx = {
                    index_set[0]: index_set[1:] if len(index_set) > 1 else []
                    for index_set in cand_indexes
                }

                # first we only mark the start_entity_idx with a prob to select an entity
                masking_indices_candidates = torch.tensor(
                    [
                        self.entity_probability if i in entity_idx.keys() else 0
                        for i in range(len(input_tokens))
                    ]
                )
                masking_indices_candidates = torch.bernoulli(masking_indices_candidates).bool()

                # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
                bernoulli = torch.bernoulli(
                    torch.full((1, len(input_tokens)), 0.8)
                ).bool()
                # get the 0/1 for the start_entity_idx positions determined by bernoulli
                indices_to_mask = (bernoulli & masking_indices_candidates).squeeze(0)

                # set the rest of an entity mask_label the same as the start_entity_idx position
                for entity_idx_start in entity_idx.keys():
                    if indices_to_mask[entity_idx_start]:
                        if len(entity_idx[entity_idx_start]) > 0:
                            indices_to_mask[entity_idx[entity_idx_start]] = True

                    # We do that for all the candidates (even those not chosen for masking)
                    if masking_indices_candidates[entity_idx_start]:  # if this is selected
                        if len(entity_idx[entity_idx_start]) > 0:
                            masking_indices_candidates[entity_idx[entity_idx_start]] = True

                # in WEP we dont have random token replacement
                # print(masking_indices_candidates)
                return (
                    masking_indices_candidates.tolist(),
                    indices_to_mask.tolist(),
                    [False] * len(input_tokens),
                    lang_tag_indices,
                    sentences_not_used,
                    cand_indexes,
                )

        else:
            sentences_not_used[self.id2lang[language]] = (
                sentences_not_used.get(self.id2lang[language], 0) + 1
            )
            return (
                [False] * len(input_tokens),
                [False] * len(input_tokens),
                [False] * len(input_tokens),
                lang_tag_indices,
                sentences_not_used,
                cand_indexes,
            )

    def torch_mask_tokens(
        self,
        batch_input,
        batch_masking_indices_candidates_entities,
        batch_mask_indices_entities,
        batch_random_indices_entities,
        batch_masking_indices_candidates_non_entities,
        batch_mask_indices_non_entities,
        batch_random_indices_non_entities,
    ):

        if self.tokenizer.mask_token is None:
            raise ValueError(
                "This tokenizer does not have a mask token which is necessary for masked language modeling. "
                "Remove the --mlm flag if you want to use this tokenizer."
            )
        labels = batch_input.clone()
        # We only compute loss on masked tokens

        # for WEP only, batch_non_entities will be False so would not contribute anyway
        # For *EP-MLM, we keep all candidates : entity and non-entity
        if self.keep_same:
            masking_candidates = (
                batch_masking_indices_candidates_entities
                | batch_masking_indices_candidates_non_entities
            )
            # for ii, jj in zip(batch_masking_indices_candidates_entities, batch_masking_indices_candidates_non_entities):
            #     for i, j in zip(ii, jj):
            #         logger.info(f'{i}\t{j}')
            # exit(0)
        else:
            # if we do not calculate loss on unchanged tokens, then we set those labels to -100 as well
            # meaning only Random and Masking matter
            masking_candidates = (
                batch_mask_indices_entities
                | batch_random_indices_entities
                | batch_masking_indices_candidates_non_entities
            )

        labels[~masking_candidates] = -100

        # will always mask entities
        batch_input[batch_mask_indices_entities] = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.mask_token
        )
        random_words = torch.randint(
            len(self.tokenizer), labels.shape, dtype=torch.long
        )

        batch_input[batch_random_indices_entities] = random_words[
            batch_random_indices_entities
        ]

        # optional to further mask tokens outside entities
        if self.masking == "ep-mlm":

            batch_input[
                batch_mask_indices_non_entities
            ] = self.tokenizer.convert_tokens_to_ids(self.tokenizer.mask_token)

            batch_input[batch_random_indices_non_entities] = random_words[
                batch_random_indices_non_entities
            ]

        return batch_input, labels
