# Copyright (C) 2024. Huawei Technologies Co., Ltd. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ============================================================================


import os
from typing import Iterable, Dict
import json
import string
import gzip



def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    https://github.com/openai/human-eval/blob/master/human_eval/data.py
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, 'rt') as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def write_jsonl(filename: str, data: Iterable[Dict], append: bool = False):
    """
    Writes an iterable of dictionaries to jsonl
    https://github.com/openai/human-eval/blob/master/human_eval/data.py
    """
    if append:
        mode = 'ab'
    else:
        mode = 'wb'
    filename = os.path.expanduser(filename)
    if filename.endswith(".gz"):
        with open(filename, mode) as fp:
            with gzip.GzipFile(fileobj=fp, mode='wb') as gzfp:
                for x in data:
                    gzfp.write((json.dumps(x) + "\n").encode('utf-8'))
    else:
        with open(filename, mode) as fp:
            for x in data:
                fp.write((json.dumps(x) + "\n").encode('utf-8'))


def read_problems(filename, infer_incremental_completions=False):
    problems = {}
    
    for intask in stream_jsonl(filename):
        outtask = {}

        for key in intask:
            if key == "signature":
                outtask["signature"] = intask[key]
            elif key == "task_id" or key == "question_id":
                outtask["task_id"] = intask[key]
            elif key == "text" and ('prompt' not in intask):
                outtask["prompt"] = intask[key]
                outtask["prompt"] = '\n'.join([lin.strip(' ') for lin in outtask["prompt"].split('\n')])
            elif key == "prompt":
                # We interpret prompt as the problem definition, and try to extract the signature from it if detected
                if "signature" in intask:
                    outtask["prompt"] = intask[key]
                else:
                    cblock = ''
                    if '"""' in intask[key]:
                        cblock = '"""'
                    elif "'''" in intask[key]:
                        cblock = "'''"

                    if cblock:
                        cblock_end = intask[key].rfind(cblock)
                        cblock_start = intask[key].rfind(cblock, 0, cblock_end)
                        prompt = intask[key][cblock_start + 3:cblock_end]
                        outtask["prompt"] = '\n'.join([lin.strip(' ') for lin in prompt.split('\n')])

                        # Strip any remaining comment blocks to avoid extracting a signature from there
                        rmd = intask[key][:cblock_start] + intask[key][cblock_end + 3:]
                        while cblock in rmd:
                            cblock_end = rmd.rfind(cblock)
                            cblock_start = rmd.rfind(cblock, 0, cblock_end)
                            rmd = rmd[:cblock_start] + rmd[cblock_end + 3:]
                        rmd = "\n".join([lin[:lin.find('#')].strip() if '#' in lin else lin.strip() for lin in rmd.split("\n")])

                        sign_start = rmd.rfind('def ')
                        args_end = rmd.find(')', sign_start)
                        sign_end = rmd.find(':', args_end)
                        outtask["signature"] = rmd[sign_start:sign_end + 1].strip()
                    else:
                        print(f'WARNING: Cannot easily distinguish problem definition from signature in the prompt in task ID {outtask["task_id"]}. Skipped!')

        # If signature was not found explicitly and not as part of the prompt, we attempt to detect it at the start of the code
        if "signature" not in outtask:
            if "code" in intask:
                sign_start = intask["code"].find('def ')
                args_end = intask["code"].find(')', sign_start)
                sign_end = intask["code"].find(':', args_end)
                outtask["signature"] = intask["code"][sign_start:sign_end + 1].strip()
            if "signature" not in outtask:
                print(f'WARNING: Cannot easily distinguish signature in code in task ID {outtask["task_id"]} or code is not provided. Skipped!')

        # Remove return typing from signature, to be consistent with training data
        if '->' in outtask["signature"]:
            idxs = outtask["signature"].find('->')
            idxe = outtask["signature"].find(':', idxs)
            outtask["signature"] = outtask["signature"][:idxs] + outtask["signature"][idxe:]
            while ' :' in outtask["signature"]:
                outtask["signature"] = outtask["signature"].replace(' :', ':')

        if infer_incremental_completions and ("canonical_solution" in intask or "code" in intask):
            if "canonical_solution" in intask:
                code = intask["canonical_solution"]
            else:
                code = intask["code"]
            if code.startswith(outtask["signature"]):
                code = code[len(outtask["signature"]):]

            # Check to avoid duplicate completion tasks due to empty lines or other artifacts
            unique_codes = set()

            code_lines = code.split('\n')
            incremental_codes = []
            incr_code = '<NEW_LINE>'
            current_indent = 0
            stored_empty_lines = 0
            for code_lin in code_lines:
                if code_lin.strip():
                    new_indent = next(i for i, j in enumerate(code_lin) if j not in string.whitespace)
                    if new_indent > current_indent:
                        while stored_empty_lines > 0:
                            incr_code += '<NEW_LINE>'
                            stored_empty_lines -= 1

                        incr_code += '<INDENT>'
                    elif new_indent < current_indent:
                        incr_code += '<DEDENT>'

                        while stored_empty_lines > 0:
                            incr_code += '<NEW_LINE>'
                            stored_empty_lines -= 1
                    else:
                        while stored_empty_lines > 0:
                            incr_code += '<NEW_LINE>'
                            stored_empty_lines -= 1
                    current_indent = new_indent

                    incr_code += code_lin.strip() + '<NEW_LINE>'
                    unique_code = incr_code[:].replace('\n', ' ').replace('<NEW_LINE>', '').strip()
                    if unique_code not in unique_codes:
                        unique_codes.add(unique_code)
                        incremental_codes.append(incr_code)
                else:
                    stored_empty_lines += 1

            completions = []
            # This is the case where we have no completion and just the signature
            completions.append("")
            # We ignore the last incremental as it contains the whole solution, i.e. nothing left to complete
            completions.extend(incremental_codes[:-1])

            for idx, c in enumerate(completions):
                dup_outtask = outtask.copy()
                dup_outtask["task_id"] = f'{dup_outtask["task_id"]}_{idx}'                
                dup_outtask["completion"] = c
                assert "task_id" in dup_outtask and "prompt" in dup_outtask and "signature" in dup_outtask and "completion" in dup_outtask
                problems[dup_outtask["task_id"]] = dup_outtask

        else:
            if infer_incremental_completions:
                print(f'WARNING: Cannot do incremental completion tasks when no solution is provided for task ID {outtask["task_id"]}. Incremental completion skipped!')
            outtask["completion"] = ""

            assert "task_id" in outtask and "prompt" in outtask and "signature" in outtask and "completion" in outtask
            problems[outtask["task_id"]] = outtask
    return problems
