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
import sys
import re
import random
import argparse
import json
import time
import shelve
from functools import reduce
from wikimapper import WikiMapper

sys.path.insert(0, "../")
from cs_pretrain.utils import humanized_time


with open("../cs_pretrain/languages/xlmr_langs.txt") as infile:
    XLMR_LANGS = [line.rstrip() for line in infile]


# [[hyperlinks|text]]
empty_hyperlinks_and_text = 0
empty_text = 0
num_entities_by_lang = {lang: 0 for lang in XLMR_LANGS}
total_hyperlinks = 0
num_sents_by_lang = {lang: 0 for lang in XLMR_LANGS}


def get_cs_sent_dict(sent, args):
    global empty_hyperlinks_and_text, empty_text, num_entities_by_lang, total_hyperlinks
    """
    sent: cleaned wikipedia sentences with [[hyperlink]] and [[hyperlink|text]]
    return: {'en': EN sent, 'lang1': lang1 sent, ...}
    """
    hyperlink_pos = []  #  [(107, 122), (131, 143), (148, 158)]
    for s, e in find_balanced(sent, ["[["], ["]]"]):
        hyperlink_pos.append((s, e))

    total_hyperlinks += len(hyperlink_pos)
    wikidata_ids = get_wikidata_ids(args, hyperlink_pos, sent)  # ['Q585', None, 'Q123']
    target_langs = get_target_langs(
        wikidata_ids, args
    )  # ['be', 'fi', 'nl', 'hu', 'ha']

    # will always have the EN sentence
    cs_sentences = {"en": ""}
    cur = 0
    for i, (s, e) in enumerate(hyperlink_pos):
        inner = sent[s + 2 : e - 2]
        pipe = inner.find("|")
        en_display_text = inner if pipe == -1 else inner[pipe + 1 :]

        if len(en_display_text) == 0:
            if pipe == -1:
                # [[]] empty, so there is no wiki_item or en_sent_text
                cs_sentences["en"] += sent[cur:s]
                empty_hyperlinks_and_text += 1
            else:
                # [[wiki_item|]], have wiki_item but no text - usually removed by {{template}} -> keep EN wiki_item
                en_wiki_item = inner[:pipe]
                if len(en_wiki_item) > 0:
                    cs_sentences["en"] += sent[cur:s] + "<en>" + en_wiki_item + "</en>"
                    empty_text += 1  # replace with en_wiki_item
                else:
                    cs_sentences["en"] += sent[cur:s]
                    empty_hyperlinks_and_text += 1
        else:
            cs_sentences["en"] += sent[cur:s] + "<en>" + en_display_text + "</en>"
        for lang in target_langs:
            if wikidata_ids[i] is not None:
                # should code-switch this entity
                num_entities_by_lang[lang] += 1
                entity_lang = args.WIKIDATA_DB[wikidata_ids[i]][lang]
                cs_sentences[lang] = (
                    cs_sentences.get(lang, "")
                    + sent[cur:s]
                    + f"<{lang}>"
                    + entity_lang
                    + f"</{lang}>"
                )
            else:
                # append the EN sentence with EN entity - Also need to check whether end up having empty <en></en>
                # cs_sentences[lang] = cs_sentences.get(lang, '') + sent[cur:s] + '<en>' + en_sent_text + '</en>'
                if len(en_display_text) == 0:
                    if pipe == -1:
                        # [[]] empty, so there is no wiki_item or en_sent_text
                        cs_sentences[lang] = cs_sentences.get(lang, "") + sent[cur:s]
                    else:
                        # [[wiki_item|]], have wiki_item but no text - usually removed by {{template}} -> keep EN wiki_item
                        en_wiki_item = inner[:pipe]
                        if len(en_wiki_item) > 0:
                            num_entities_by_lang["en"] += 1
                            cs_sentences[lang] = (
                                cs_sentences.get(lang, "")
                                + sent[cur:s]
                                + "<en>"
                                + en_wiki_item
                                + "</en>"
                            )
                        else:
                            cs_sentences[lang] = (
                                cs_sentences.get(lang, "") + sent[cur:s]
                            )
                else:
                    num_entities_by_lang["en"] += 1
                    cs_sentences[lang] = (
                        cs_sentences.get(lang, "")
                        + sent[cur:s]
                        + "<en>"
                        + en_display_text
                        + "</en>"
                    )

        cur = e
    for key in cs_sentences.keys():
        # original sentences are with \n at the end, remove in the sentence dict
        cs_sentences[key] += sent[cur:-1]
    return cs_sentences


def get_wikidata_ids(args, hyperlink_pos, sent):
    """
    return list of wikidata_id that will cs (None if not cs)
    this depends on (1) if a sentence has <= 3 hyperlinks, cs all
                    (2) if a sentence has > 3 hyperlinks, cs args.switch_percent determine by random number
                    (3) if a hyperlink cannot find wikidata id, then do not consider this as it will result in
                       0 shared target languages then no hyperlinks will get cs
    """
    wikidata_ids = []
    for pos in hyperlink_pos:
        # wikidata_id = None
        # if len(hyperlink_pos) <= 3 or random.random() <= args.switch_percent:
        # we are switching all entities now
        wikidata_id = get_wikidata_id(pos, sent, args)
        if wikidata_id is not None:
            if wikidata_id not in args.WIKIDATA_DB:
                # only save the valid wikidata_id that can be found in WIKIDATA_DB
                wikidata_id = None
        wikidata_ids.append(wikidata_id)
    return wikidata_ids


def get_target_langs(wikidata_ids, args):
    shared_target_langs = []  # list of set of available languages for a wikidata item
    for wikidata_id in wikidata_ids:
        if wikidata_id is not None:
            available_langs = set(
                [
                    lang
                    for lang in args.WIKIDATA_DB[wikidata_id].keys()
                    if lang in XLMR_LANGS
                ]
            )
            # we do not include EN as target_lang
            if "en" in available_langs:
                available_langs.remove("en")
            shared_target_langs.append(available_langs)
    if len(shared_target_langs) > 0:
        shared_target_langs = list(
            reduce(lambda i, j: i & j, (x for x in shared_target_langs))
        )
    if len(shared_target_langs) > args.num_translations:
        shared_target_langs = random.sample(
            shared_target_langs, k=args.num_translations
        )
    return shared_target_langs


def get_wikidata_id(pos, sent, args):
    """
    return the page title that is saved in wikimapper (from wikipedia redirect sql)
    the title is case sensitive and all spaces are replaced by _
    but the text in the article sentences might not follow the case as saved in the wikimapper
    therefore check hyperlink in the sentences whether exist (1) as it is (aquatic_fern) NO
                                                             (2) capitalised first letter of each word (Aquatic_Fern) NO
                                                             (3) capitalised first letter of fist word (Aquatic_fern) YES
    then save the wikidata id if can find, otherwise return None
    """

    inner = sent[pos[0] + 2 : pos[1] - 2]
    if len(inner) == 0:
        return None
    pipe = inner.find("|")
    # case like [[|KsheHalachta]]
    if pipe == 0:
        return None
    hyperlink_text = (
        inner.replace(" ", "_") if pipe == -1 else inner[:pipe].replace(" ", "_")
    )
    if hyperlink_text.startswith("[["):
        if len(hyperlink_text) > 2:
            hyperlink_text = hyperlink_text[2:]
        else:
            return None
    # for cases like '[Atharvaveda|AtharvaVeda'
    if hyperlink_text.startswith("["):
        if len(hyperlink_text) > 2:
            hyperlink_text = hyperlink_text[1:]
        else:
            return None
    wikidata_id = args.WIKI_MAPPER.title_to_id(hyperlink_text)
    if wikidata_id is not None:
        return wikidata_id
    # str.title() will convert all letters after the first letter in each word to lower case, double check if issue
    wikidata_id = args.WIKI_MAPPER.title_to_id(hyperlink_text.title())
    if wikidata_id is not None:
        return wikidata_id
    if len(hyperlink_text) > 1:
        wikidata_id = args.WIKI_MAPPER.title_to_id(
            hyperlink_text[0].upper() + hyperlink_text[1:]
        )
    else:
        wikidata_id = args.WIKI_MAPPER.title_to_id(hyperlink_text[0].upper())
    if wikidata_id is not None:
        return wikidata_id
    return None


def find_balanced(text, open_delim, close_delim):
    """
    Assuming that text contains a properly balanced expression using
    :param open_delim: as opening delimiters and
    :param close_delim: as closing delimiters.
    :return: an iterator producing pairs (start, end) of start and end
    positions in text containing a balanced expression.
    """
    open_pat = "|".join([re.escape(x) for x in open_delim])
    # patter for delimiters expected after each opening delimiter
    after_pat = {
        o: re.compile(open_pat + "|" + c, re.DOTALL)
        for o, c in zip(open_delim, close_delim)
    }
    stack = []
    start = 0
    cur = 0
    # end = len(text)
    start_set = False
    start_pat = re.compile(open_pat)
    next_pat = start_pat
    while True:
        next = next_pat.search(text, cur)
        if not next:
            return
        if not start_set:
            start = next.start()
            start_set = True
        delim = next.group(0)
        if delim in open_delim:
            stack.append(delim)
            next_pat = after_pat[delim]
        else:
            opening = stack.pop()
            # assert opening == open_delim[close_delim.index(next.group(0))]
            if stack:
                next_pat = after_pat[stack[-1]]
            else:
                yield start, next.end()
                next_pat = start_pat
                start = next.end()
                start_set = False
        cur = next.end()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--sent_folder",
        type=str,
        default="clean_sentences_latest",
        help="folder for EN sentences",
    )
    parser.add_argument(
        "--cs_folder", type=str, help="folder for saving code switched sentences"
    )
    parser.add_argument(
        "--switch_percent",
        type=float,
        default=1.0,
        help="percentage of the entities for code-switching",
    )
    parser.add_argument(
        "--num_translations",
        type=int,
        default=5,
        help="num of translations for code-switched entities",
    )
    args = parser.parse_args()

    if args.cs_folder is None:
        args.cs_folder = "data_by_lang_v7"
    os.makedirs(args.cs_folder, exist_ok=False)

    print("loading wikidata db!")
    t_0 = time.time()
    args.WIKIDATA_DB = shelve.open("wikidata_db/wikidata_db")
    print(f"wikidata db loaded. Time Elapsed {humanized_time(time.time() - t_0)}.")
    print("wikidata_db loaded!")
    args.WIKI_MAPPER = WikiMapper("wikimapper_data/index_enwiki-latest.db")

    with open("clean_sent_in_one.txt", "r") as file:
        for i, line in enumerate(file):
            if i % 1000000 == 0:
                print(f"{i} sentences processed.")
            cs_sent_dict = get_cs_sent_dict(line, args)
            en_sentence = cs_sent_dict["en"]
            with open("en_sentences_v7.json", "a") as en_file:
                num_sents_by_lang["en"] += 1
                en_line_dict = {"id": i, "en_sentence": en_sentence}
                en_file.write(json.dumps(en_line_dict, ensure_ascii=False) + "\n")
            for lang, cs_sentence in cs_sent_dict.items():
                if not lang == "en":
                    num_sents_by_lang[lang] += 1
                    parallel_line_dict = {
                        "id": i,
                        "en_sentence": en_sentence,
                        "cs_sentence": cs_sentence,
                        "language": lang,
                    }
                    with open(
                        f"{args.cs_folder}/cs_sents_en_{lang}.json", "a"
                    ) as cs_parallel_file:
                        cs_parallel_file.write(
                            json.dumps(parallel_line_dict, ensure_ascii=False) + "\n"
                        )


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"All done! Time elapsed {humanized_time(time.time() - t0)}.")
    print(f"empty_hyperlinks_and_text: {empty_hyperlinks_and_text}")
    print(f"empty_text_in [[]] which is replaced with en_wiki_item: {empty_text}")
    print(f"total_hyperlinks: {total_hyperlinks}")
    with open("num_entities_by_lang_v7.json", "w") as fp:
        json.dump(num_entities_by_lang, fp, indent=4)
    with open("num_sents_by_lang_v7.json", "w") as fp:
        json.dump(num_sents_by_lang, fp, indent=4)
