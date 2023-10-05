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


import datasets
import os


def main():
    langs = "ar,he,vi,id,jv,ms,tl,eu,ml,ta,te,af,nl,en,de,el,bn,hi,mr,ur,fa,fr,it,pt,es,bg,ru,ja,ka,ko,th,sw,yo,my,zh,kk,tr,et,fi,hu"
    langs = langs.split(',')
    original_dataset = {}

    if not os.path.exists('../data/'):
        os.makedirs('../data/')

    for l in langs:
        original_dataset[l] = datasets.load_dataset('xtreme', f'PAN-X.{l}')
        print(original_dataset[l])
        original_dataset[l].save_to_disk(f"../data/wikiann/{l}")


if __name__ == "__main__":
    main()
