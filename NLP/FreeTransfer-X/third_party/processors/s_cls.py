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
from .utils import InputExample

logger = logging.getLogger(__name__)


class SingleCLSProcessor(DataProcessor):
  """Processor for the single-sentence-classification dataset.
  Adapted from https://github.com/google-research/bert/blob/f39e881b169b9d53bea03d2d341b31707a6c052b/run_classifier.py#L207"""

  def __init__(self):
    pass

  def get_examples(self, data_dir, language='en', split='train'):
    """See base class."""
    examples = []
    for lg in language.split(','):
      lines = self._read_tsv(os.path.join(data_dir, split, "{}".format(lg))) # 210728: my file struct
      for (i, line) in enumerate(lines):
        guid = "%s-%s-%s" % (split, lg, i)
        text_a = line[0]
        label = line[1].strip()
        assert isinstance(text_a, str) and isinstance(label, str)
        examples.append(InputExample(guid=guid, text_a=text_a, label=label, language=lg))
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


class MtopSClsProcessor(SingleCLSProcessor):
  def get_labels(self):
    return ['IN:ADD_TIME_TIMER', 'IN:ADD_TO_PLAYLIST_MUSIC', 'IN:ANSWER_CALL', 'IN:CANCEL_CALL', 'IN:CANCEL_MESSAGE', 'IN:CREATE_ALARM', 'IN:CREATE_CALL', 'IN:CREATE_PLAYLIST_MUSIC', 'IN:CREATE_REMINDER', 'IN:CREATE_TIMER', 'IN:DELETE_ALARM', 'IN:DELETE_PLAYLIST_MUSIC', 'IN:DELETE_REMINDER', 'IN:DELETE_TIMER', 'IN:DISLIKE_MUSIC', 'IN:DISPREFER', 'IN:END_CALL', 'IN:FAST_FORWARD_MUSIC', 'IN:FOLLOW_MUSIC', 'IN:GET_AGE', 'IN:GET_AIRQUALITY', 'IN:GET_ALARM', 'IN:GET_ATTENDEE_EVENT', 'IN:GET_AVAILABILITY', 'IN:GET_CALL', 'IN:GET_CALL_CONTACT', 'IN:GET_CALL_TIME', 'IN:GET_CATEGORY_EVENT', 'IN:GET_CONTACT', 'IN:GET_CONTACT_METHOD', 'IN:GET_DATE_TIME_EVENT', 'IN:GET_DETAILS_NEWS', 'IN:GET_EDUCATION_DEGREE', 'IN:GET_EDUCATION_TIME', 'IN:GET_EMPLOYER', 'IN:GET_EMPLOYMENT_TIME', 'IN:GET_EVENT', 'IN:GET_GENDER', 'IN:GET_GROUP', 'IN:GET_INFO_CONTACT', 'IN:GET_INFO_RECIPES', 'IN:GET_JOB', 'IN:GET_LANGUAGE', 'IN:GET_LIFE_EVENT', 'IN:GET_LIFE_EVENT_TIME', 'IN:GET_LOCATION', 'IN:GET_LYRICS_MUSIC', 'IN:GET_MAJOR', 'IN:GET_MESSAGE', 'IN:GET_MESSAGE_CONTACT', 'IN:GET_MUTUAL_FRIENDS', 'IN:GET_RECIPES', 'IN:GET_REMINDER', 'IN:GET_REMINDER_AMOUNT', 'IN:GET_REMINDER_DATE_TIME', 'IN:GET_REMINDER_LOCATION', 'IN:GET_STORIES_NEWS', 'IN:GET_SUNRISE', 'IN:GET_SUNSET', 'IN:GET_TIMER', 'IN:GET_TRACK_INFO_MUSIC', 'IN:GET_UNDERGRAD', 'IN:GET_WEATHER', 'IN:HELP_REMINDER', 'IN:HOLD_CALL', 'IN:IGNORE_CALL', 'IN:IS_TRUE_RECIPES', 'IN:LIKE_MUSIC', 'IN:LOOP_MUSIC', 'IN:MERGE_CALL', 'IN:PAUSE_MUSIC', 'IN:PAUSE_TIMER', 'IN:PLAY_MEDIA', 'IN:PLAY_MUSIC', 'IN:PREFER', 'IN:PREVIOUS_TRACK_MUSIC', 'IN:QUESTION_MUSIC', 'IN:QUESTION_NEWS', 'IN:REMOVE_FROM_PLAYLIST_MUSIC', 'IN:REPEAT_ALL_MUSIC', 'IN:REPEAT_ALL_OFF_MUSIC', 'IN:REPLAY_MUSIC', 'IN:RESTART_TIMER', 'IN:RESUME_CALL', 'IN:RESUME_MUSIC', 'IN:RESUME_TIMER', 'IN:REWIND_MUSIC', 'IN:SEND_MESSAGE', 'IN:SET_AVAILABLE', 'IN:SET_DEFAULT_PROVIDER_CALLING', 'IN:SET_DEFAULT_PROVIDER_MUSIC', 'IN:SET_RSVP_INTERESTED', 'IN:SET_RSVP_NO', 'IN:SET_RSVP_YES', 'IN:SET_UNAVAILABLE', 'IN:SHARE_EVENT', 'IN:SILENCE_ALARM', 'IN:SKIP_TRACK_MUSIC', 'IN:SNOOZE_ALARM', 'IN:START_SHUFFLE_MUSIC', 'IN:STOP_MUSIC', 'IN:STOP_SHUFFLE_MUSIC', 'IN:SUBTRACT_TIME_TIMER', 'IN:SWITCH_CALL', 'IN:UNLOOP_MUSIC', 'IN:UPDATE_ALARM', 'IN:UPDATE_CALL', 'IN:UPDATE_METHOD_CALL', 'IN:UPDATE_REMINDER', 'IN:UPDATE_REMINDER_DATE_TIME', 'IN:UPDATE_REMINDER_LOCATION', 'IN:UPDATE_REMINDER_TODO', 'IN:UPDATE_TIMER']


class MatisSClsProcessor(SingleCLSProcessor):
  def get_examples(self, data_dir, language='en', split='train'):
    """See base class."""
    examples = []
    for lg in language.split(','):
      lines = self._read_tsv(os.path.join(data_dir, split, "{}".format(lg))) # 210728: my file struct
      for (i, line) in enumerate(lines):
        guid = "%s-%s-%s" % (split, lg, i)
        text_a = line[0]
        label = line[1].strip()
        label = re.split(r"#| ", label)[0] # NOTE
        assert isinstance(text_a, str) and isinstance(label, str)
        examples.append(InputExample(guid=guid, text_a=text_a, label=label, language=lg))
    return examples

  def get_labels(self):
    return ['atis_abbreviation', 'atis_aircraft', 'atis_airfare', 'atis_airline', 'atis_airport', 'atis_capacity', 'atis_cheapest', 'atis_city', 'atis_day_name', 'atis_distance', 'atis_flight', 'atis_flight_no', 'atis_flight_time', 'atis_ground_fare', 'atis_ground_service', 'atis_meal', 'atis_quantity', 'atis_restriction']



mtop_processors = {
  "mtop": MtopSClsProcessor,
}

matis_processors = {
  "m_atis": MatisSClsProcessor,
}

mtop_output_modes = {
  "mtop": "classification",
}

matis_output_modes = {
  "m_atis": "classification",
}

mtop_tasks_num_labels = {
  "mtop": 113, # FIXME: its paper say 117 intents
}

matis_tasks_num_labels = {
  "m_atis": 18,
}
