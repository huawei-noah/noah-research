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


class Server(Sandbox):
    def __init__(self, capacity, hourly_service_rate):
        super().__init__()
        self.capacity = capacity
        self.hourly_service_rate = hourly_service_rate
        self.number_pending = self.add_hour_counter()
        self.number_in_service = self.add_hour_counter()
        self.on_start = Action()

    def request_to_start(self):
        self.number_pending.observe_change(1)
        print("{0}\t{1}\tRequestToStart. #Pending: {2}. #In-Service: {3}".format(self.clock_time, type(self).__name__,
                                                                                 self.number_pending.last_count,
                                                                                 self.number_in_service.last_count))
        if self.number_in_service.last_count < self.capacity:
            self.start()

    def start(self):
        self.number_pending.observe_change(-1)
        self.number_in_service.observe_change(1)
        print("{0}\t{1}\tStart. #Pending: {2}. #In-Service: {3}".format(self.clock_time, type(self).__name__,
                                                                        self.number_pending.last_count,
                                                                        self.number_in_service.last_count))
        self.schedule([self.finish], timedelta(hours=round(random.expovariate(self.hourly_service_rate), 2)))
        self.on_start.invoke()

    def finish(self):
        self.number_in_service.observe_change(1)
        print("{0}\t{1}\tStart. #Pending: {2}. #In_Service: {3}".format(self.clock_time, type(self).__name__,
                                                                        self.number_pending.last_count,
                                                                        self.number_in_service.last_count))
        if self.number_pending.last_count > 0:
            self.start()

