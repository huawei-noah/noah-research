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


class Event:
    __count = 0

    def __init__(self, owner, action, scheduled_time, tag=None):
        Event.__count += 1
        self.__index = Event.__count
        self.__owner = owner
        self.__action = action
        self.__scheduled_time = scheduled_time
        self.__tag = tag

    @property
    def index(self):
        return self.__index

    @property
    def owner(self):
        return self.__owner

    @property
    def action(self):
        return self.__action

    @property
    def scheduled_time(self):
        return self.__scheduled_time

    @property
    def tag(self):
        return self.__tag

    def invoke(self):
        if len(self.__action) == 1:
            self.__action[0]()
        else:
            self.__action[0](**self.__action[1])

    def __str__(self):
        return f'{self.__tag}#{self.__index}'

    def __eq__(self, other):
        if isinstance(other, Event):
            return self.__scheduled_time == other.scheduled_time and self.__index == other.index
        else:
            raise TypeError()

    def __lt__(self, other):
        if isinstance(other, Event):
            if self.__scheduled_time == other.scheduled_time:
                return self.__index < other.index
            else:
                return self.__scheduled_time < other.scheduled_time
        else:
            raise TypeError()

    def __hash__(self):
        return hash(self.__index)
