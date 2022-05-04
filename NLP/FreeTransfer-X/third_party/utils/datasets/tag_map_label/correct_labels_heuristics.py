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

'''m_atis requires BIO correction
  mtop does NOT
'''

import os
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--in_fmt_label_data", type=str, help="tsv, phrase-level formatted real data with true labels") # NOTE: e.g. `original/m_atis-seq_tag/test/zh`
parser.add_argument("--out_fmt_label_data", type=str, help="tsv. w/ heuristically corrected labels")
args = parser.parse_args()

LABEL_LIST = ['I-meal_description', 'B-airport_name', 'I-toloc.airport_code', 'B-arrive_date.date_relative', 'I-toloc.city_name', 'B-flight_number', 'B-toloc.airport_code', 'I-period_of_day', 'B-time_relative', 'I-aircraft_code', 'I-connect', 'I-return_date.date_relative', 'B-fromloc.city_name', 'I-arrive_date.day_name', 'B-flight_mod', 'B-days_code', 'I-fromloc.airport_code', 'B-month_name', 'I-today_relative', 'O', 'I-depart_time.time_relative', 'B-airport_code', 'B-depart_date.day_name', 'I-depart_date.date_relative', 'I-arrive_time.time', 'B-depart_date.today_relative', 'B-meal_code', 'I-cost_relative', 'I-state_name', 'B-day_number', 'I-fromloc.state_code', 'I-round_trip', 'I-restriction_code', 'B-stoploc.airport_name', 'I-time', 'B-meal', 'B-fromloc.state_name', 'B-fare_amount', 'I-depart_time.time', 'I-fromloc.airport_name', 'I-class_type', 'I-loc.state_name', 'B-depart_date.month_name', 'B-transport_type', 'B-return_date.month_name', 'I-or', 'I-fromloc.state_name', 'B-restriction_code', 'B-class_type', 'B-arrive_time.period_of_day', 'I-fare_basis_code', 'B-arrive_time.period_mod', 'I-days_code', 'B-arrive_date.month_name', 'B-connect', 'I-meal_code', 'I-arrive_time.start_time', 'B-today_relative', 'I-flight_number', 'I-arrive_time.period_mod', 'B-round_trip', 'I-transport_type', 'B-period_of_day', 'I-arrive_date.date_relative', 'B-arrive_time.time_relative', 'I-compartment', 'I-airport_name', 'I-depart_time.period_mod', 'I-arrive_date.month_name', 'B-depart_date.day_number', 'I-airline_name', 'B-return_date.date_relative', 'B-flight_stop', 'B-stoploc.city_name', 'I-arrive_date.day_number', 'B-stoploc.airport_code', 'B-depart_time.start_time', 'B-arrive_date.day_number', 'I-time_relative', 'B-airline_name', 'B-toloc.state_name', 'B-or', 'I-flight_days', 'B-toloc.country_name', 'B-toloc.airport_name', 'B-return_time.period_mod', 'B-booking_class', 'I-fromloc.city_name', 'I-airport_code', 'I-depart_date.day_name', 'I-toloc.state_code', 'B-arrive_date.day_name', 'I-return_date.day_number', 'B-fromloc.airport_code', 'B-economy', 'B-airline_code', 'B-fare_basis_code', 'I-return_date.today_relative', 'B-stoploc.state_code', 'B-state_code', 'I-stoploc.city_name', 'B-meal_description', 'B-arrive_time.start_time', 'B-return_time.period_of_day', 'B-time', 'B-fromloc.state_code', 'B-depart_time.period_of_day', 'I-depart_date.month_name', 'B-toloc.state_code', 'B-cost_relative', 'I-mod', 'B-depart_time.period_mod', 'B-return_date.today_relative', 'B-depart_date.date_relative', 'B-toloc.city_name', 'I-return_time.period_of_day', 'B-state_name', 'I-flight_stop', 'I-arrive_time.end_time', 'B-return_date.day_name', 'B-day_name', 'B-compartment', 'I-toloc.airport_name', 'I-depart_date.today_relative', 'I-arrive_time.period_of_day', 'I-fare_amount', 'B-flight', 'B-fromloc.airport_name', 'B-flight_days', 'B-depart_date.year', 'I-depart_date.year', 'I-city_name', 'I-depart_date.day_number', 'I-flight_time', 'B-flight_time', 'I-flight_mod', 'I-depart_time.start_time', 'I-airline_code', 'I-economy', 'B-depart_time.time', 'I-depart_time.period_of_day', 'B-mod', 'I-toloc.state_name', 'B-aircraft_code', 'B-city_name', 'B-arrive_time.end_time', 'B-arrive_time.time', 'I-arrive_time.time_relative', 'B-arrive_date.today_relative', 'B-depart_time.time_relative', 'B-return_date.day_number', 'I-depart_time.end_time', 'B-depart_time.end_time']


def heuristics(labels):
  '''update continuous identical labels to obey BIO rules, 
    1) the first should be B-
    2) following should be I-
    e.g. 
      1) O,I-x -> O,B-x
      2) I-x,I-x -> B-x,I-x
      3) I-x,B-x -> B-x,I-x
  '''
  def norm_label(label):
    if label == 'O':
      return label
    else:
      return label.split('-')[1]

  def read_stack(label_stack):
    new_labels = []
    for i, nlab in enumerate(label_stack):
      if nlab == 'O':
        new_labels.append(nlab)
      else:
        if i == 0:
          if f"B-{nlab}" in LABEL_LIST:
            new_labels.append(f"B-{nlab}")
          else:
            new_labels.append(f"I-{nlab}")
        else:
          if f"I-{nlab}" in LABEL_LIST:
            new_labels.append(f"I-{nlab}")
          else:
            new_labels.append(f"B-{nlab}")
    return new_labels

  new_labels = []
  label_stack = []
  for lab in labels:
    norm_lab = norm_label(lab)
    if label_stack:
      if label_stack[-1] == norm_lab:
        label_stack.append(norm_lab)
      else: # to clean stack
        new_labels.extend(read_stack(label_stack))
        label_stack = [norm_lab]
    else: # empty stack
      label_stack.append(norm_lab)
  if label_stack: new_labels.extend(read_stack(label_stack))
  return new_labels


def main():
  dirname = os.path.dirname(args.out_fmt_label_data)
  os.makedirs(dirname, exist_ok=True)

  sent_toks, sent_labs = [], []
  with open(args.in_fmt_label_data, 'r') as fin, open(args.out_fmt_label_data, 'w') as fout:
    for line in fin:
      line = line.strip()
      if line:
        tok, lab = line.split('\t')
        sent_toks.append(tok)
        sent_labs.append(lab)
      else:
        nsent_labs = heuristics(sent_labs)
        assert len(sent_toks) == len(nsent_labs), f"len(sent_toks) = {len(sent_toks)}, len(nsent_labs) = {len(nsent_labs)}, len(sent_labs) = {len(sent_labs)}"
        for tok, lab in zip(sent_toks, nsent_labs):
          fout.write(f"{tok}\t{lab}\n")
        fout.write('\n')
        sent_toks, sent_labs = [], []


if __name__ == "__main__":
  main()
