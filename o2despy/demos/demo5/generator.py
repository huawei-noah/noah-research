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
from o2despy.action import Action
from datetime import timedelta
import random


class Generator(Sandbox):
    def __init__(self, hourly_rate):
        super().__init__()
        self.hourly_rate = hourly_rate
        self.count = self.add_hour_counter()
        self.on_generate = Action()

        self.schedule([self.generate])

    def generate(self):
        if self.count.last_count > 0:
            print("{0}\t{1}\tGenerate. Count: {2}".format(self.clock_time, type(self).__name__, self.count.last_count))
            self.on_generate.invoke()
        self.count.observe_change(1)
        self.schedule([self.generate], timedelta(hours=round(random.expovariate(self.hourly_rate), 2)))
