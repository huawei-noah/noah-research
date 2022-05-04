# coding=utf-8
# Copyright 2018 The Google AI Language Team Authors and The HuggingFace Inc. team.
# Copyright (c) 2018, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
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


import logging
import os
import re

from transformers import DataProcessor
from .utils_tag import read_examples_from_file

logger = logging.getLogger(__name__)


class TagProcessor(DataProcessor):
  """Processor for the single-sentence-classification dataset.
  Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

  def __init__(self):
    self.lang2id = None

  def set_lang2id(self, lang2id):
    self.lang2id = lang2id

  def get_examples(self, data_dir, language='en', split='train'):
    """See base class."""
    examples = []
    for lg in language.split(','):
      file_path = os.path.join(data_dir, split, "{}".format(lg)) # 211203
      examples.extend(read_examples_from_file(file_path, lg, self.lang2id))
    return examples

  def get_train_examples(self, data_dir, language='en'):
    return self.get_examples(data_dir, language, split='train')

  def get_dev_examples(self, data_dir, language='en'):
    return self.get_examples(data_dir, language, split='dev')

  def get_test_examples(self, data_dir, language='en'):
    return self.get_examples(data_dir, language, split='test')

  def get_translate_train_examples(self, data_dir, language='en'):
    """See base class."""
    return self.get_examples(data_dir, language, split='trans-train') # 211011

  def get_translate_test_examples(self, data_dir, language='en'):
    return self.get_examples(data_dir, language, split='trans-test') # 211011
    
  def get_pseudo_test_examples(self, data_dir, language='en'):
    return self.get_examples(data_dir, language, split='pseudo_test') # 211011

  def get_aug_train_examples(self, data_dir, language='en'):
    return self.get_examples(data_dir, language, split='aug-train')

  def get_labels(self):
    raise NotImplementedError()

  def get_labels_map(self):
    '''210731: skip `get_labels` to adapt to `transformers-4.9.1`'''
    label_list = self.get_labels()
    label2id = {label:i for i, label in enumerate(label_list)}
    id2label = {i:label for i, label in enumerate(label_list)}
    return (label2id, id2label)


class MtopTagProcessor(TagProcessor):
  def get_labels(self):
    return ['CONTACT_METHOD', 'TYPE_RELATION', 'MUSIC_PLAYLIST_MODIFIER', 'MUSIC_ALBUM_TITLE', 'MUSIC_ARTIST_NAME', 'RECIPES_COOKING_METHOD', 'TIMER_NAME', 'RECIPES_UNIT_NUTRITION', 'EMPLOYER', 'AMOUNT', 'SENDER', 'NEWS_TOPIC', 'ATTENDEE_EVENT', 'ATTRIBUTE_EVENT', 'NAME_APP', 'MUSIC_GENRE', 'GENDER', 'RECIPES_DIET', 'AGE', 'RECIPIENT', 'ALARM_NAME', 'RECIPES_CUISINE', 'TODO', 'RECIPES_EXCLUDED_INGREDIENT', 'CONTACT_ADDED', 'EDUCATION_DEGREE', 'METHOD_RETRIEVAL_REMINDER', 'LIFE_EVENT', 'MUSIC_PLAYLIST_TITLE', 'GROUP', 'RECIPES_TIME_PREPARATION', 'PERSON_REMINDED', 'NEWS_CATEGORY', 'USER_ATTENDEE_EVENT', 'MUSIC_TYPE', 'TYPE_CONTACT', 'RECIPES_MEAL', 'MUSIC_PROVIDER_NAME', 'NEWS_REFERENCE', 'METHOD_TIMER', 'CONTENT_EXACT', 'O', 'SCHOOL', 'METHOD_RECIPES', 'WEATHER_ATTRIBUTE', 'RECIPES_QUALIFIER_NUTRITION', 'NEWS_TYPE', 'SIMILARITY', 'MUSIC_RADIO_ID', 'TITLE_EVENT', 'ORDINAL', 'MUSIC_ALBUM_MODIFIER', 'JOB', 'MUSIC_REWIND_TIME', 'RECIPES_TYPE_NUTRITION', 'WEATHER_TEMPERATURE_UNIT', 'NEWS_SOURCE', 'RECIPES_UNIT_MEASUREMENT', 'MUSIC_TRACK_TITLE', 'CONTACT_REMOVED', 'RECIPES_RATING', 'MAJOR', 'DATE_TIME', 'RECIPES_TYPE', 'LOCATION', 'PHONE_NUMBER', 'ATTENDEE', 'PERIOD', 'RECIPES_ATTRIBUTE', 'TYPE_CONTENT', 'RECIPES_SOURCE', 'RECIPES_DISH', 'CATEGORY_EVENT', 'RECIPES_INCLUDED_INGREDIENT', 'CONTACT_RELATED', 'CONTACT']


class MatisTagProcessor(TagProcessor):
  def get_labels(self):
    return ['I-meal_description', 'B-airport_name', 'I-toloc.airport_code', 'B-arrive_date.date_relative', 'I-toloc.city_name', 'B-flight_number', 'B-toloc.airport_code', 'I-period_of_day', 'B-time_relative', 'I-aircraft_code', 'I-connect', 'I-return_date.date_relative', 'B-fromloc.city_name', 'I-arrive_date.day_name', 'B-flight_mod', 'B-days_code', 'I-fromloc.airport_code', 'B-month_name', 'I-today_relative', 'O', 'I-depart_time.time_relative', 'B-airport_code', 'B-depart_date.day_name', 'I-depart_date.date_relative', 'I-arrive_time.time', 'B-depart_date.today_relative', 'B-meal_code', 'I-cost_relative', 'I-state_name', 'B-day_number', 'I-fromloc.state_code', 'I-round_trip', 'I-restriction_code', 'B-stoploc.airport_name', 'I-time', 'B-meal', 'B-fromloc.state_name', 'B-fare_amount', 'I-depart_time.time', 'I-fromloc.airport_name', 'I-class_type', 'I-loc.state_name', 'B-depart_date.month_name', 'B-transport_type', 'B-return_date.month_name', 'I-or', 'I-fromloc.state_name', 'B-restriction_code', 'B-class_type', 'B-arrive_time.period_of_day', 'I-fare_basis_code', 'B-arrive_time.period_mod', 'I-days_code', 'B-arrive_date.month_name', 'B-connect', 'I-meal_code', 'I-arrive_time.start_time', 'B-today_relative', 'I-flight_number', 'I-arrive_time.period_mod', 'B-round_trip', 'I-transport_type', 'B-period_of_day', 'I-arrive_date.date_relative', 'B-arrive_time.time_relative', 'I-compartment', 'I-airport_name', 'I-depart_time.period_mod', 'I-arrive_date.month_name', 'B-depart_date.day_number', 'I-airline_name', 'B-return_date.date_relative', 'B-flight_stop', 'B-stoploc.city_name', 'I-arrive_date.day_number', 'B-stoploc.airport_code', 'B-depart_time.start_time', 'B-arrive_date.day_number', 'I-time_relative', 'B-airline_name', 'B-toloc.state_name', 'B-or', 'I-flight_days', 'B-toloc.country_name', 'B-toloc.airport_name', 'B-return_time.period_mod', 'B-booking_class', 'I-fromloc.city_name', 'I-airport_code', 'I-depart_date.day_name', 'I-toloc.state_code', 'B-arrive_date.day_name', 'I-return_date.day_number', 'B-fromloc.airport_code', 'B-economy', 'B-airline_code', 'B-fare_basis_code', 'I-return_date.today_relative', 'B-stoploc.state_code', 'B-state_code', 'I-stoploc.city_name', 'B-meal_description', 'B-arrive_time.start_time', 'B-return_time.period_of_day', 'B-time', 'B-fromloc.state_code', 'B-depart_time.period_of_day', 'I-depart_date.month_name', 'B-toloc.state_code', 'B-cost_relative', 'I-mod', 'B-depart_time.period_mod', 'B-return_date.today_relative', 'B-depart_date.date_relative', 'B-toloc.city_name', 'I-return_time.period_of_day', 'B-state_name', 'I-flight_stop', 'I-arrive_time.end_time', 'B-return_date.day_name', 'B-day_name', 'B-compartment', 'I-toloc.airport_name', 'I-depart_date.today_relative', 'I-arrive_time.period_of_day', 'I-fare_amount', 'B-flight', 'B-fromloc.airport_name', 'B-flight_days', 'B-depart_date.year', 'I-depart_date.year', 'I-city_name', 'I-depart_date.day_number', 'I-flight_time', 'B-flight_time', 'I-flight_mod', 'I-depart_time.start_time', 'I-airline_code', 'I-economy', 'B-depart_time.time', 'I-depart_time.period_of_day', 'B-mod', 'I-toloc.state_name', 'B-aircraft_code', 'B-city_name', 'B-arrive_time.end_time', 'B-arrive_time.time', 'I-arrive_time.time_relative', 'B-arrive_date.today_relative', 'B-depart_time.time_relative', 'B-return_date.day_number', 'I-depart_time.end_time', 'B-depart_time.end_time']

