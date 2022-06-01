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
# THE SOFTWARE.


from functools import partial
from collections import Iterable
from utils.readonly import ReadOnlyList


class Action:
    def __init__(self, *args, **kwargs):
        self.__subactions = []
        self.__methods = []
        self.__arg_num = len(args) + len(kwargs)

    @property
    def subactions(self):
        return ReadOnlyList(self.__subactions)

    @property
    def methods(self):
        return ReadOnlyList(self.__methods)

    def invoke(self, *args, **kwargs):
        for func in self.__subactions:
            func(*args, **kwargs)

    def clear(self):
        self.__subactions.clear()
        self.__methods.clear()

    def add(self, action):
        if isinstance(action, Action):
            subactions = action.subactions
        elif isinstance(action, dict):
            subactions = action.values()
        elif isinstance(action, Iterable):
            subactions = action
        else:
            subactions = [action]
        for subaction in subactions:
            self.__add_subaction(subaction)
        return self

    def __add_subaction(self, subaction):
        if not callable(subaction):  # check if it is a method
            raise TypeError(f"'{subaction}' is not callable.")
        method, result, msg = self.__check_subaction(subaction)  # check the method args num
        if result is False:
            raise TypeError(msg)
        self.__subactions.append(subaction)
        self.__methods.append(method)

    def __check_subaction(self, subaction):
        method = subaction
        method_arg_num = 0
        if isinstance(subaction, partial):
            method = subaction.func
            method_arg_num -= len(subaction.args) + len(subaction.keywords)
        if not hasattr(method, '__name__'):
            raise TypeError(f"Unexpected type.")
        method_arg_num += method.__code__.co_argcount
        if hasattr(method, '__self__'):
            method_arg_num -= 1
        if method_arg_num == self.__arg_num or method_arg_num == -1:
            return method, True, ""
        else:
            msg = f"'{method.__name__}' has mismatched number of arguments, excepted {self.__arg_num} instead of {method_arg_num}."
            return method, False, msg

    def __add__(self, other):
        return self.add(other)

    def __len__(self):
        return len(self.__subactions)
