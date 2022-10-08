# coding=utf-8
# Copyright 2022 Huawei Technologies Co., Ltd.
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

import random
import math
import re

def remove_space_and_bracket(text):
    text = text.strip().replace(" ", "")
    text = re.sub("[\[\{]", "(", text)
    text = re.sub("[\]\}]", ")", text)
    return text

def split_eq(text):
    text = re.split(r"([=\+\-\*\/\{\}\(\)\[\]\^])", text)
    return [x for x in text if x]

def get_test_nums():
    #mapping from symbol numbers to real numbers
    nums = {"#_pi": 3.14, "PI": 3.14}
    for i in range(10):
        nums[str(i)] = float(i)
    for i in range(50):
        nums[f"#{i}"] = random.random()
    return nums

def calculate_eval(equation, nums):
    op_list = ["+", "-", "*", "/", "(", ")", "^"]
    equation = clean_text(equation).split(" ")
    try:
        for i, e in enumerate(equation):
            if e not in op_list:
                if e in nums:
                    equation[i] = str(nums[e])
                else:
                    equation[i] = e
            if equation[i][-1] == "%":
                equation[i] = f"( {equation[i][:-1]} / 100 )"

        after_number_exp = " ".join(equation)
        assert not '#' in after_number_exp
        after_number_exp = after_number_exp.replace("^", "**")
        ans = eval(after_number_exp)
    except:
        return None
    return ans

def is_equal(label, text):
    for test_times in range(3):
        failed = 0
        label_ans = None
        while label_ans is None:
            failed += 1
            if failed == 5:
                return False
            nums = get_test_nums()
            label_ans = calculate_eval(label, nums)
        text_ans = calculate_eval(text, nums)
        try:
            if text_ans is None or abs(text_ans - label_ans) > 1e-5:
                return False
        except:
            return False
    return True

def clean_text(text):
    splited_text = split_eq(remove_space_and_bracket(text))
    bracket = 0
    for i, s in enumerate(splited_text):
        if s=="(":
            bracket += 1
        elif s == ")":
            bracket -= 1
            if bracket < 0:
                return " ".join(splited_text[:i])
    return " ".join(splited_text)
