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


class MMcQueue(Sandbox):
    def __init__(self, hourly_arrival_rate, hourly_service_rate, capacity, seed=0):
        super().__init__(seed=seed)
        self.hourly_arrival_rate = hourly_arrival_rate
        self.hourly_service_rate = hourly_service_rate
        self.capacity = capacity
        self.in_queue = self.add_hour_counter()
        self.in_service = self.add_hour_counter()

        # self.schedule([self.arrive], timedelta(seconds=0))
        self.schedule([self.arrive])

    def arrive(self):
        if self.in_service.last_count < self.capacity:
            self.in_service.observe_change(1)
            print("{0}\tArrive and Start Service (In-Queue: {1}, In-Service: {2})".
                  format(self.clock_time, self.in_queue.last_count, self.in_service.last_count))
            self.schedule([self.depart], timedelta(hours=round(random.expovariate(self.hourly_service_rate), 2)))
        else:
            self.in_queue.observe_change(1)
            print("{0}\tArrive and Join Queue (In-Queue: {1}, In-Service: {2})".
                  format(self.clock_time, self.in_queue.last_count, self.in_service.last_count))
        self.schedule([self.arrive], timedelta(hours=round(random.expovariate(self.hourly_arrival_rate), 2)))

    def depart(self):
        if self.in_queue.last_count > 0:
            self.in_queue.observe_change(-1)
            print("{0}\tDepart and Start Service (In-Queue: {1}, In-Service: {2})".
                  format(self.clock_time, self.in_queue.last_count, self.in_service.last_count))
            self.schedule([self.depart], timedelta(hours=round(random.expovariate(self.hourly_service_rate), 2)))
        else:
            self.in_service.observe_change(-1)
            print("{0}\tDepart (In-Queue: {1}, In-Service: {2})".
                  format(self.clock_time, self.in_queue.last_count, self.in_service.last_count))

    
if __name__ == '__main__':
    # Demo 3
    print("Demo 3 - MMcQueue")
    sim = MMcQueue(hourly_arrival_rate=1, hourly_service_rate=2, capacity=2)
    sim.warmup(period=datetime.timedelta(hours=24))
    sim.run(duration=datetime.timedelta(hours=30))
