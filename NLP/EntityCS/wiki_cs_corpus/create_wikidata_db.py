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

from qwikidata.json_dump import WikidataJsonDump
import shelve

wjd = WikidataJsonDump("latest-all.json.bz2")


def main():
    db = shelve.open("wikidata_db")
    for i, en_dict in enumerate(wjd):
        item_dict = {}
        labels = en_dict["labels"]
        for key, value in labels.items():
            item_dict[key] = value["value"]
        db[en_dict["id"]] = item_dict
    db.close()
    print("All Done!")


if __name__ == "__main__":
    main()
