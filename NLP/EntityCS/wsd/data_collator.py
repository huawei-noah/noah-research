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
import logging
from transformers import PreTrainedTokenizerBase
from transformers.file_utils import PaddingStrategy
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union

InputDataClass = NewType("InputDataClass", Any)

"""
A DataCollator is a function that takes a list of samples from a Dataset and collate them into a batch, as a dictionary
of PyTorch/TensorFlow tensors or NumPy arrays.
"""
DataCollator = NewType("DataCollator", Callable[[List[InputDataClass]], Dict[str, Any]])


class DataCollatorMixin:
    def __call__(self, features, return_tensors=None):
        if return_tensors is None:
            return_tensors = self.return_tensors
        if return_tensors == "tf":
            return self.tf_call(features)
        elif return_tensors == "pt":
            return self.torch_call(features)
        elif return_tensors == "np":
            return self.numpy_call(features)
        else:
            raise ValueError(f"Framework '{return_tensors}' not recognized!")


def default_data_collator(features: List[InputDataClass], return_tensors="pt") -> Dict[str, Any]:
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    """

    # In this function we'll make the assumption that all `features` in the batch
    # have the same attributes.
    # So we will look at the first element as a proxy for what attributes exist
    # on the whole batch.

    if return_tensors == "pt":
        return torch_default_data_collator(features)
    elif return_tensors == "tf":
        return tf_default_data_collator(features)
    elif return_tensors == "np":
        return numpy_default_data_collator(features)


@dataclass
class DefaultDataCollator(DataCollatorMixin):
    """
    Very simple data collator that simply collates batches of dict-like objects and performs special handling for
    potential keys named:
        - `label`: handles a single value (int or float) per object
        - `label_ids`: handles a list of values per object
    Does not do any additional preprocessing: property names of the input object will be used as corresponding inputs
    to the model. See glue and ner for example of how it's useful.
    This is an object (like other data collators) rather than a pure function like default_data_collator. This can be
    helpful if you need to set a return_tensors value at initialization.
    Args:
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]], return_tensors=None) -> Dict[str, Any]:
        if return_tensors is None:
            return_tensors = self.return_tensors
        return default_data_collator(features, return_tensors)


def torch_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import torch

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], torch.Tensor) else first["label"]
        dtype = torch.long if isinstance(label, int) else torch.float
        batch["labels"] = torch.tensor([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], torch.Tensor):
            batch["labels"] = torch.stack([f["label_ids"] for f in features])
        else:
            dtype = torch.long if type(first["label_ids"][0]) is int else torch.float
            batch["labels"] = torch.tensor([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, torch.Tensor):
                batch[k] = torch.stack([f[k] for f in features])
            else:
                batch[k] = torch.tensor([f[k] for f in features])

    return batch


def tf_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import numpy as np
    import tensorflow as tf

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label_col_name = "label"
    elif "label_ids" in first and first["label_ids"] is not None:
        label_col_name = "label_ids"
    elif "labels" in first and first["labels"] is not None:
        label_col_name = "labels"
    else:
        label_col_name = None
    if label_col_name is not None:
        if isinstance(first[label_col_name], tf.Tensor):
            dtype = tf.int64 if first[label_col_name].dtype.is_integer() else tf.float32
        elif isinstance(first[label_col_name], np.ndarray) or isinstance(first[label_col_name], np.generic):
            dtype = tf.int64 if np.issubdtype(first[label_col_name].dtype, np.integer) else tf.float32
        elif isinstance(first[label_col_name], (tuple, list)):
            dtype = tf.int64 if isinstance(first[label_col_name][0], int) else tf.float32
        else:
            dtype = tf.int64 if isinstance(first[label_col_name], int) else tf.float32
        batch["labels"] = tf.convert_to_tensor([f[label_col_name] for f in features], dtype=dtype)
    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids", "labels") and v is not None and not isinstance(v, str):
            if isinstance(v, (tf.Tensor, np.ndarray)):
                batch[k] = tf.stack([f[k] for f in features])
            else:
                batch[k] = tf.convert_to_tensor([f[k] for f in features])

    return batch


def numpy_default_data_collator(features: List[InputDataClass]) -> Dict[str, Any]:
    import numpy as np

    if not isinstance(features[0], (dict, BatchEncoding)):
        features = [vars(f) for f in features]
    first = features[0]
    batch = {}

    # Special handling for labels.
    # Ensure that tensor is created with the correct type
    # (it should be automatically the case, but let's make sure of it.)
    if "label" in first and first["label"] is not None:
        label = first["label"].item() if isinstance(first["label"], np.ndarray) else first["label"]
        dtype = np.int64 if isinstance(label, int) else np.float32
        batch["labels"] = np.array([f["label"] for f in features], dtype=dtype)
    elif "label_ids" in first and first["label_ids"] is not None:
        if isinstance(first["label_ids"], np.ndarray):
            batch["labels"] = np.stack([f["label_ids"] for f in features])
        else:
            dtype = np.int64 if type(first["label_ids"][0]) is int else np.float32
            batch["labels"] = np.array([f["label_ids"] for f in features], dtype=dtype)

    # Handling of all other possible keys.
    # Again, we will use the first element to figure out which key/values are not None for this model.
    for k, v in first.items():
        if k not in ("label", "label_ids") and v is not None and not isinstance(v, str):
            if isinstance(v, np.ndarray):
                batch[k] = np.stack([f[k] for f in features])
            else:
                batch[k] = np.array([f[k] for f in features])

    return batch


@dataclass
class NewDataCollatorWithPadding:
    """
    Data collator that will dynamically pad the inputs received.
    Args:
        tokenizer ([`PreTrainedTokenizer`] or [`PreTrainedTokenizerFast`]):
            The tokenizer used for encoding the data.
        padding (`bool`, `str` or [`~file_utils.PaddingStrategy`], *optional*, defaults to `True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:
            - `True` or `'longest'`: Pad to the longest sequence in the batch (or no padding if only a single sequence
              if provided).
            - `'max_length'`: Pad to a maximum length specified with the argument `max_length` or to the maximum
              acceptable input length for the model if that argument is not provided.
            - `False` or `'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of different
              lengths).
        max_length (`int`, *optional*):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (`int`, *optional*):
            If set will pad the sequence to a multiple of the provided value.
            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        return_tensors (`str`):
            The type of Tensor to return. Allowable values are "np", "pt" and "tf".
    """

    tokenizer: PreTrainedTokenizerBase
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    return_tensors: str = "pt"

    def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
        batched = {}
        # print(features)
        batched['input_ids'] = _torch_collate_batch([f['input_ids'] for f in features], self.tokenizer)
        batched['attention_mask'] = torch.nn.utils.rnn.pad_sequence([torch.tensor(f['attention_mask']) for f in features],
                                                                    batch_first=True,
                                                                    padding_value=0)
        # batched['target_tokens_1'] = torch.tensor([f['target_tokens_1'] for f in features])
        # batched['target_tokens_2'] = torch.tensor([f['target_tokens_2'] for f in features])
        batched['target_tokens_1'] = torch.tensor([f['target_tokens_1'] for f in features])
        # torch.nn.utils.rnn.pad_sequence([torch.tensor(f['target_tokens_1']) for f in features],
        #                                                              batch_first=True,
        #                                                              padding_value=0)
        batched['target_tokens_2'] = torch.tensor([f['target_tokens_2'] for f in features])
        # torch.nn.utils.rnn.pad_sequence(
        # [torch.tensor(f['target_tokens_2']) for f in features],
        #                                                          batch_first=True,
        #                                                          padding_value=0)

        batched["labels"] = torch.tensor([f["labels"] for f in features]).long()
        return batched


def _torch_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    import numpy as np
    import torch

    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple, np.ndarray)):
        examples = [torch.tensor(e, dtype=torch.long) for e in examples]

    length_of_first = examples[0].size(0)

    # Check if padding is necessary.

    are_tensors_same_length = all(x.size(0) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
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


def _tf_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    import numpy as np
    import tensorflow as tf

    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [tf.convert_to_tensor(e, dtype=tf.int64) for e in examples]

    # Check if padding is necessary.
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return tf.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    # result = examples[0].new_full([len(examples), max_length], tokenizer.pad_token_id)
    result = []
    rank = tf.rank(examples[0])
    paddings = np.zeros((rank, 2), dtype=np.int32)
    for example in examples:
        if tokenizer.padding_side == "right":
            paddings[0, 1] = max_length - len(example)
        else:
            paddings[0, 0] = max_length - len(example)
        result.append(tf.pad(example, paddings, constant_values=tokenizer.pad_token_id))
    return tf.stack(result, axis=0)


def _numpy_collate_batch(examples, tokenizer, pad_to_multiple_of: Optional[int] = None):
    import numpy as np

    """Collate `examples` into a batch, using the information in `tokenizer` for padding if necessary."""
    # Tensorize if necessary.
    if isinstance(examples[0], (list, tuple)):
        examples = [np.array(e, dtype=np.int64) for e in examples]

    # Check if padding is necessary.
    length_of_first = len(examples[0])
    are_tensors_same_length = all(len(x) == length_of_first for x in examples)
    if are_tensors_same_length and (pad_to_multiple_of is None or length_of_first % pad_to_multiple_of == 0):
        return np.stack(examples, axis=0)

    # If yes, check if we have a `pad_token`.
    if tokenizer._pad_token is None:
        raise ValueError(
            "You are attempting to pad samples but the tokenizer you are using"
            f" ({tokenizer.__class__.__name__}) does not have a pad token."
        )

    # Creating the full tensor and filling it with our data.
    max_length = max(len(x) for x in examples)
    if pad_to_multiple_of is not None and (max_length % pad_to_multiple_of != 0):
        max_length = ((max_length // pad_to_multiple_of) + 1) * pad_to_multiple_of
    result = np.full(shape=(len(examples), max_length), fill_value=tokenizer.pad_token_id, dtype=examples[0].dtype)
    for i, example in enumerate(examples):
        if tokenizer.padding_side == "right":
            result[i, : example.shape[0]] = example
        else:
            result[i, -example.shape[0] :] = example
    return result


def tolist(x):
    if isinstance(x, list):
        return x
    elif hasattr(x, "numpy"):  # Checks for TF tensors without needing the import
        x = x.numpy()
    return x.tolist()


