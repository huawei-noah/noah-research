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

import os
import time
import sys

sys.path.insert(0, "../")
from utils import humanized_time


def main():
    count = 0
    sent_folder = "clean_sentences_latest"
    sentence_files = os.listdir(sent_folder)
    ave_sent_len = 0
    total_sentences = 0

    for sentence_file in sentence_files:
        print(f"Processing file {sentence_file}")
        with open(os.path.join(sent_folder, sentence_file), "r") as file:
            for i, line in enumerate(file):
                if "rowspan=" in line or "colspan=" in line or "style=" in line:
                    continue
                if "[[File:" in line or "[[Image:" in line:
                    continue
                length = len(line.split())
                if length > 128:
                    count += 1
                    continue
                else:
                    ave_sent_len = (ave_sent_len * total_sentences + length) / (
                        total_sentences + 1
                    )
                    total_sentences += 1
                    with open("clean_sent_in_one.txt", "a") as one_file:
                        if line[-1] == "\n":
                            one_file.write(line)
                        else:
                            one_file.write(line + "\n")

    print(f"All done! Removed {count} sentences that is longer than 128.")
    print(f"total_sentences: {total_sentences}, ave_sent_len: {ave_sent_len}.")


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"Time elapsed {humanized_time(time.time() - t0)}.")
