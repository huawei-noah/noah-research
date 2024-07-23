# Copyright (C) 2020 EleutherAI

from typing import Iterable
from lm_eval.base import BaseLM
from mindnlp.transformers import OPTTokenizer, OPTForCausalLM
import mindspore
import torch


class MindSporeLM(BaseLM):

	DEFAULT_MAX_LENGTH: int = 2048

	def __init__(self, pretrained, batch_size: int = 1, device: str = 'cpu'):
		super().__init__()
		self.model = OPTForCausalLM.from_pretrained(pretrained)
		self.tokenizer = OPTTokenizer.from_pretrained(pretrained)
		self._batch_size = int(batch_size)
		self._device = device

	@property
	def max_length(self):
		seqlen_config_attrs = ("n_positions", "max_position_embeddings", "n_ctx")
		for attr in seqlen_config_attrs:
			if hasattr(self.model.config, attr):
				return getattr(self.model.config, attr)
		return self.DEFAULT_MAX_LENGTH

	@property
	def max_gen_toks(self):
		raise NotImplementedError

	@property
	def batch_size(self):
		return self._batch_size

	@property
	def device(self):
		return self._device

	def tok_encode(self, string: str):
		return self.tokenizer.encode(string).ids

	def tok_decode(self, tokens: Iterable[int]):
		raise NotImplementedError

	def _model_generate(self, context, max_length, eos_token_id):
		raise NotImplementedError

	def _model_call(self, inps):
		inps = mindspore.Tensor(inps.numpy())  # hack
		return torch.tensor(self.model(inps)[0].asnumpy())

	@property
	def eot_token_id(self):
		return self.tokenizer.eos_token_id
