#!/bin/bash

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

folder=$1
mkdir "${folder}/mldoc_corpus"

for lang in 'chinese' 'german' 'japanese' 'spanish' 'french' 'italian' 'russian'; do

  python generate_documents.py --indices-file "${folder}/mldoc-indices/${lang}.dev" \
                               --output-filename "${folder}/mldoc_corpus/${lang}.dev" \
                               --rcv-dir RCV2_Multilingual_Corpus/${lang}

  python generate_documents.py --indices-file "${folder}/mldoc-indices/${lang}.test" \
                               --output-filename "${folder}/mldoc_corpus/${lang}.test" \
                               --rcv-dir "${folder}/RCV2_Multilingual_Corpus/${lang}"
done

# English
python generate_documents.py --indices-file "${folder}/mldoc-indices/english.train.1000" \
                             --output-filename "${folder}/mldoc_corpus/en.train.1000" \
                             --rcv-dir "${folder}/rcv1"

python generate_documents.py --indices-file "${folder}/mldoc-indices/english.dev" \
                             --output-filename "${folder}/mldoc_corpus/en.dev" \
                             --rcv-dir "${folder}/rcv1"

python generate_documents.py --indices-file "${folder}/mldoc-indices/english.test" \
                             --output-filename "${folder}/mldoc_corpus/en.test" \
                             --rcv-dir "${folder}/rcv1"
