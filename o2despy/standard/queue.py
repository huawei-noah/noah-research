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


from o2despy.sandbox import ISandbox, Sandbox
from abc import ABCMeta, abstractclassmethod


class IQueue(ISandbox):
    @property
    def capacity(self):
        return self.__capacity

    @property
    def pending_to_enqueue(self):
        return self.__pending_to_enqueue
    
    @property
    def queueing(self):
        return self.__queueing
    
    @property
    def occupancy(self):
        return self.__occupancy
    
    @property
    def vacancy(self):
        return self.__vacancy
    
    @property
    def utilization(self):
        return self.__utilization
    
    @property
    def avgn_queueing(self):
        return self.__avgn_queueing

    @property
    def on_enqueued(self):
        return self.__on_enqueued
    
    @abstractclassmethod
    def rqst_enqueue(self, load):
        pass

    @abstractclassmethod
    def dequeue(self, load):
        pass


class Queue(Sandbox, IQueue):
    def __init__(self, capacity, seed=0, id=None):
        super().__init__(seed=seed, identifier=id)
        self.__pending_to_enqueue = []
        self.__queueing = []
        self.__hc_queueing = self.add_hour_counter()
        self.__on_enqueued = []
        self.__capacity = capacity

    @property
    def capacity(self):
        return self.__capacity

    @property
    def pending_to_enqueue(self):
        return self.__pending_to_enqueue
    
    @property
    def queueing(self):
        return self.__queueing
    
    @property
    def occupancy(self):
        return len(self.queueing)
    
    @property
    def vacancy(self):
        return self.capacity - self.occupancy
    
    @property
    def utilization(self):
        return self.avgn_queueing / self.capacity
    
    @property
    def avgn_queueing(self):
        return self.__hc_queueing.average_count

    def rqst_enqueue(self, load):
        self.__pending_to_enqueue.append(load)
        self.atmpt_enqueue()
    
    def dequeue(self, load):
        if load in self.__queueing:
            self.__queueing.remove(load)
            self.__hc_queueing.observe_change(-1, self.clock_time)
            self.atmpt_enqueue()

    def atmpt_enqueue(self):
        if len(self.__pending_to_enqueue) > 0 and len(self.__queueing) < self.capacity:
            load = self.__pending_to_enqueue[0]
            self.__queueing.append(load)
            self.__pending_to_enqueue.pop(0)
            self.__hc_queueing.observe_change(1, self.clock_time)
            for func in self.__on_enqueued:
                if len(func) == 1:
                    func[0](load)
                else:
                    func[0](**func[1])

    @property
    def on_enqueued(self):
        return self.__on_enqueued

    @on_enqueued.setter
    def on_enqueued(self, value):
        self.__on_enqueued.append(value)

