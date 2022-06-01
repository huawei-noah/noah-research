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


from o2despy.sandbox import Sandbox
import random


class Queue(Sandbox):
    def __init__(self):
        super().__init__()
        self.number_waiting = self.add_hour_counter()

    def enqueue(self):
        self.number_waiting.observe_change(1)
        print("{0}\t{1}\tEnqueue. #Waiting: {2}".format(self.clock_time,
                                                        type(self).__name__,
                                                        self.number_waiting.last_count))

    def dequeue(self):
        self.number_waiting.observe_change(-1)
        print("{0}\t{1}\tDequeue. #Waiting: {2}".format(self.clock_time,
                                                        type(self).__name__,
                                                        self.number_waiting.last_count))
