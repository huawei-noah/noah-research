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


def replaceInternalLinksCustomised(text):
    """
    three cases: (1) remove [[...]] e.g. [[Category:...]] or [[:Category:...]]
                 (2) replace: [[unwanted title|label]], remove unwanted title, remove [[]]. replace label:
                     - [[File: ...|..|label]], [[:File: ...|..|label]], [[Image:...]], [[w]], [[wiktionary]], [[wikt]]
                     - [[title#..|label]] -> cannot  find wikidata id anyway
                 (3) keep [[title]] and [[title|label]]
    [[title |...|label]]trail

    with title concatenated with trail, when present, e.g. 's' for plural.
    """
    # call this after removal of external links, so we need not worry about
    # triple closing ]]].
    text = text.replace('&amp;', '&')
    cur = 0
    res = ''
    for s, e in findBalanced(text, ['[['], [']]']):
        m = tailRE.match(text, e)
        if m:
            trail = m.group(0)
            end = m.end()
        else:
            trail = ''
            end = e
        inner = text[s + 2:e - 2]
        # find first |
        pipe = inner.find('|')
        normal_hyperlink = is_normal_hyperlink(inner)

        if pipe < 0:
            title = inner
            label = title
        else:
            title = inner[:pipe].rstrip()
            # find last |
            curp = pipe + 1
            for s1, e1 in findBalanced(inner, ['[['], [']]']):
                last = inner.rfind('|', curp, s1)
                if last >= 0:
                    pipe = last  # advance
                curp = e1
            label = inner[pipe + 1:].strip()
        if inner.startswith('Category:') or inner.startswith(':Category:'):
            # remove [[Category:...]] and [[:Category:...]] completely
            res += text[cur:s] + trail
        else:
            if normal_hyperlink:
                res += text[cur:s] + '[[' + inner + ']]' + trail
            else:
                res += text[cur:s] + makeInternalLink(title, label) + trail
        cur = end
    return res + text[cur:]


def is_normal_hyperlink(inner_hyperlink):
    num_pipe = inner_hyperlink.count('|')
    if num_pipe > 1:
        return False
    if inner_hyperlink.startswith('File:') or inner_hyperlink.startswith('Image:'):
        return False
    if num_pipe == 0:
        return True
    pipe = inner_hyperlink.find('|')
    if '#' in inner_hyperlink[:pipe] or ':' in inner_hyperlink[:pipe]:
        return False
    return True