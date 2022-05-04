#!/usr/bin/env python3
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
import sys
import argparse

import torch

from transformers import M2M100ForConditionalGeneration, M2M100Tokenizer


class M2M100(object):

  def __init__(self, model_path):
    self._model = M2M100ForConditionalGeneration.from_pretrained(model_path)
    self._tokenizer = M2M100Tokenizer.from_pretrained(model_path)
    self._fids = []
    self._dim = ""

    self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    self._model.to(self._device)

  def set_src_lang(self, src_lang):
    self._tokenizer.src_lang = src_lang

  def translate(self, text, tgt_lang, do_sample=False, num_return_sequences=1, temperature=1.0):
    # TODO: to improve efficiency, batch translation?
    encoded = self._tokenizer(text, return_tensors="pt").to(self._device)
    generated_tokens = self._model.generate(**encoded, forced_bos_token_id=self._tokenizer.get_lang_id(tgt_lang), 
        do_sample=do_sample, num_return_sequences=num_return_sequences, temperature=temperature,
        ) # prompt-based generation, `transformers.generation_utils`
    return self._tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)

  def set_fields(self, fids, dim):
    self._fids = [int(fid) for fid in fids]
    self._dim = dim

  def translate_fields(self, text, tgt_lang, do_sample=False, num_return_sequences=1, temperature=1.0):
    '''return: 
        trans = [
          "{trans_field[0]\ttrans_field[1]\t...}",
          "{trans_field[0]\ttrans_field[1]\t...}",
          ...
        ]
    '''
    fields = text.split(self._dim)
    trans = []
    for i, field in enumerate(fields):
      trans_field = []
      if i in self._fids: 
        trans_field = self.translate(field.strip(), tgt_lang, 
            do_sample=do_sample, num_return_sequences=num_return_sequences, temperature=temperature)
      else: 
        trans_field = [field.strip()] * num_return_sequences

      for j in range(num_return_sequences): # 211213
        if not trans_field[j].strip():
          print(f" [translate_fields] WARNING: empty elements {j}, trans_field = [{trans_field}]")
          sys.stdout.flush()
          break
      if trans:
        for j in range(num_return_sequences):
          trans[j] = self._dim.join([trans[j], trans_field[j]])
      else: # the 1st field
        trans.extend(trans_field)

    # return self._dim.join(trans)
    return trans


def test():
  '''test the translation'''
  chkpt = "."
  model = M2M100(chkpt)

  text = "जीवन एक चॉकलेट बॉक्स की तरह है।"
  model.set_src_lang("hi")
  print(model.translate(text, "en"))

  text = "生活就像一盒巧克力。"
  model.set_src_lang("zh")
  print(model.translate(text, "en"))

if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("--src", type=str, help="")
  parser.add_argument("--tgt", type=str, default="en", help="")
  parser.add_argument("--text", type=str, help="")
  parser.add_argument("--in_files", type=str, default="", help="")
  parser.add_argument("--out_dir", type=str, default="", help="")
  parser.add_argument("--model_dir", type=str, help="")
  parser.add_argument("--test", action="store_true", help="")
  parser.add_argument("--fields", type=str, default="", help="field ids, comma-split & 0-start")
  parser.add_argument("--fields_dim", type=str, default="\t", help="")
  args = parser.parse_args()

  if args.test: test()

  model = M2M100(args.model_dir)
  model.set_src_lang(args.src)
  if args.fields:
    model.set_fields(args.fields.split(','), args.fields_dim)

  if args.in_files:
    os.makedirs(args.out_dir, exist_ok=True)
    for path in args.in_files.split(','):
      filename = os.path.basename(path)
      out_path = os.path.join(args.out_dir, filename + f'.{args.tgt}')
      with open(path, 'r') as fin, open(out_path, 'w') as fout:
        if args.fields: 
          for line in fin: fout.write(model.translate_fields(line.strip(), args.tgt) + '\n')
        else: 
          for line in fin: fout.write(model.translate(line.strip(), args.tgt) + '\n')
  elif args.text:
    if args.fields: print(model.translate_fields(args.text, args.tgt))
    else: print(model.translate(args.text, args.tgt))
  else:
    raise TypeError(f"{args.text}, {args.in_files}") 
