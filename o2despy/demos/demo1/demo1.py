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


import datetime
import random
from datetime import timedelta
from o2despy.sandbox import Sandbox


class HelloWorld(Sandbox):
    def __init__(self, hourly_arrival_rate, seed=0):
        super().__init__(seed=seed)
        self.hourly_arrival_rate = hourly_arrival_rate
        self.hc = self.add_hour_counter()

        # self.schedule([self.arrive], timedelta(seconds=0))
        self.schedule([self.arrive])

    def arrive(self):
        self.hc.observe_change(1)
        print("{}\tHello World #{}! Cum Value: {}".format(self.clock_time, self.hc.last_count, self.hc.cum_value))
        self.schedule([self.arrive], timedelta(hours=round(random.expovariate(1 / self.hourly_arrival_rate), 2)))


if __name__ == '__main__':
    # Demo 1
    print("Demo 1 - Hello world")
    sim = HelloWorld(2, seed=1)
    sim.warmup(period=datetime.timedelta(hours=24))
    sim.run(duration=datetime.timedelta(hours=30))
