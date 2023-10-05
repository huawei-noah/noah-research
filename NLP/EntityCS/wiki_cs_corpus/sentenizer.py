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

"""
take Wikidump text that has been preprocessed by the modified WikiExtractor, split to sentences with hyperlinks
"""

import time
import os
import argparse
import json
import re
from spacy.language import Language
import spacy


MIN_SEN_LENGTH = 20
ONE_LINK_MAX_PROPORTION = 0.5
nlp = spacy.load("en_core_web_sm")


@Language.component("set_custom_boundaries")
def set_custom_boundaries(doc):
    for token in doc[:-1]:
        if token.text == "[":
            doc[token.i + 1].is_sent_start = False
        if (
            token.text == '''"''' and doc[token.i + 1].text == "["
        ):  # avoid splitting "[[
            doc[token.i + 1].is_sent_start = False
        if (
            token.text == "]]" and doc[token.i + 1].text == "["
        ):  # avoid splitting "]] [[
            doc[token.i + 1].is_sent_start = False
    return doc


nlp.add_pipe("set_custom_boundaries", before="parser")


def get_clean_sentences(file_name, final_file_name, stats_dict):
    """
    remove lines with <doc>, </doc> and titles (without [[]])
    there are several cases spacy doesnt sentenize well:
    1) sentences might end with '[[' which should be the start of the next line --> remove this
    2) '[[' might be missing from a line even it is not at the last line --> if there is no starting [[ but
        ending ]] at the 'start', we add '[[' at the beginning
    """
    with open(file_name, "r") as file, open(final_file_name, "a") as cleanfile:
        lines = file.readlines()
        for line in lines:
            if (
                "<doc>" not in line
                and "</doc>" not in line
                and "[[" in line
                and "]]" in line
            ):
                doc = nlp(line)
                for sent in doc.sents:
                    # here spacy might have removed starting '[[' from a sentence then it won't be considered as
                    # sentences with hyperlinks, so first add the missing '[['
                    cur_sent = str(sent.text)

                    if cur_sent[-1] == "\n":
                        # remove ending \n, since will all add later
                        cur_sent = cur_sent[:-1]
                    if cur_sent[-2:] == "[[":
                        # remove the ending '[[' from a sentence
                        cur_sent = cur_sent[:-2]
                    if ignore_sentence(cur_sent):
                        continue
                    else:
                        if "[[" not in cur_sent or cur_sent.find("]]") < cur_sent.find(
                            "[["
                        ):
                            cur_sent = "[[" + cur_sent
                        # after adding starting [[, check whether matching the ignore pattern again
                        if ignore_sentence(cur_sent):
                            continue
                        if "[[" in sent.text and "]]" in sent.text:
                            num_hyperlinks = int(cur_sent.count("[["))
                            # only keep sentences with <= 10 hyperlinks
                            # here remove the [[Bourne shell]] (sh) and [[Business computing]]
                            # if there is only one hyperlink and the inner length > 50% total length
                            if num_hyperlinks == 1 and ignore_sentence(
                                cur_sent, one_link=True
                            ):
                                continue
                            if num_hyperlinks <= 10:
                                cur_sent = replace_symbols(cur_sent)
                                stats_dict["num_hyperlinks"][num_hyperlinks] = (
                                    stats_dict["num_hyperlinks"].get(num_hyperlinks, 0)
                                    + 1
                                )
                                if stats_dict["ave_sent_len"] == 0:
                                    stats_dict["ave_sent_len"] = len(cur_sent.split())
                                else:
                                    stats_dict["ave_sent_len"] = (
                                        stats_dict["ave_sent_len"]
                                        * stats_dict["total_sentences"]
                                        + len(cur_sent.split())
                                    ) / (stats_dict["total_sentences"] + 1)
                                stats_dict["total_sentences"] = (
                                    stats_dict.get("total_sentences", 0) + 1
                                )
                                cleanfile.write(cur_sent + "\n")
    return stats_dict


def replace_symbols(sent):
    sent = sent.replace("&amp;", "&")
    sent = sent.replace("&lt;br&gt;", "")
    sent = sent.replace("&lt;ref&gt;", "")
    sent = sent.replace("&lt;/ref&gt;", "")
    sent = sent.replace('&lt;/ref"&gt;', "")
    sent = sent.replace("&lt;ref", "")
    sent = sent.replace("&lt;br", "")
    sent = sent.replace("&lt;", "")
    sent = sent.replace("&gt;", "")
    # remove extra spaces
    sent = re.sub(" +", " ", sent)
    sent = sent.replace("( ; ) ", "")
    sent = sent.replace("(, )", "")
    sent = sent.replace(", ;", ",")
    sent = sent.replace("( ; ", "(")
    sent = sent.replace("(; ", "(")
    sent = sent.replace("(;", "(")
    sent = sent.replace("(; )", "")
    sent = sent.replace("()", "")
    sent = sent.replace("( )", "")
    sent = re.sub(" +", " ", sent)
    return sent


def ignore_sentence(sent, one_link=None):
    """
    ignore sentences with the following patterns
    '! scope="col" and '! scope="row"
    there are cases with 'style="position:sticky;top:0;" rowspan="3"
    """
    if one_link is None:
        if "rowspan=" in sent or "colspan=" in sent or "style=" in sent:
            return True
        if sent.startswith('''scope="row"''') or sent.startswith('''scope="col"'''):
            return True
        if sent.startswith("Coach: [["):
            return True
        if "]]" not in sent:
            return True
        if len(sent) <= MIN_SEN_LENGTH:
            return True
        if sent.startswith("[[") and sent.endswith("]] –"):  # [[100VG-AnyLAN]] –
            return True
        if sent.startswith("[[") and sent.endswith(") –"):  # [[Standard ML]] (or SML) –
            return True
    else:
        if one_link:
            # calculate ->[[...]]<- length
            hyperlink_length = sent.find("]]") - sent.find("[[") + 2
            if hyperlink_length / len(sent) > ONE_LINK_MAX_PROPORTION:
                return True
    return False


def main(args):
    stats_dict = {"num_hyperlinks": {}, "ave_sent_len": 0}
    file_list = os.listdir(args.wiki_folder)
    os.makedirs(args.result_folder, exist_ok=True)
    sent_file = os.path.join(
        args.result_folder, f"{args.wiki_folder.split('/')[-1]}.txt"
    )
    start = time.time()
    for file in file_list:
        stats_dict = get_clean_sentences(
            os.path.join(args.wiki_folder, file), sent_file, stats_dict
        )
    print(
        f"Done! Folder {args.wiki_folder.split('/')[-1]} with {len(file_list)} files processed, this took {time.time() - start} seconds."
    )
    stats_folder = os.path.join(args.result_folder, "stats")
    os.makedirs(stats_folder, exist_ok=True)
    with open(
        os.path.join(stats_folder, f"{args.wiki_folder.split('/')[-1]}.json"), "w+"
    ) as fp:
        json.dump(stats_dict, fp, indent=4)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--wiki_folder", type=str, help="folder for extracting sentences"
    )
    parser.add_argument(
        "--result_folder",
        type=str,
        default="clean_sentences_latest",
        help="folder for saving extracted sentences",
    )
    args = parser.parse_args()
    wiki_all_dir = "wikiextractor/text"
    folder_list = os.listdir(wiki_all_dir)
    for folder in folder_list:
        args.wiki_folder = os.path.join(wiki_all_dir, folder)
        print(f"start to process folder {args.wiki_folder}")
        main(args)
