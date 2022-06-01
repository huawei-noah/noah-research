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
import datetime
import random


class PingPongPlayer(Sandbox):
    def __init__(self, index, delay_time_expected, delay_time_CV, seed=0):
        super().__init__(seed=seed)
        self.index = index
        self.delay_time_expected = delay_time_expected
        self.delay_time_CV = delay_time_CV
        self.count = self.add_hour_counter()
        # self.on_send = []
        self.on_send = Action()

    @property
    def index(self):
        return self.__index

    @index.setter
    def index(self, value):
        self.__index = value

    def send(self):
        print(f"{self.clock_time}\t Send. Player #{self.index}, Count: {self.count.last_count}")
        # for func in self.on_send:
        #     if len(func) == 1:
        #         func[0]()
        #     else:
        #         func[0](**func[1])
        self.on_send.invoke()

    def receive(self):
        self.count.observe_change(1)
        print(f"{self.clock_time}\t Receive. Player #{self.index}, Count: {self.count.last_count}")
        self.schedule([self.send], datetime.timedelta(seconds=round(random.gammavariate(self.delay_time_expected, self.delay_time_CV),2)))
