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
from o2despy.assets import IAssets
from abc import ABCMeta, abstractclassmethod


class IServer(ISandbox):
    @property
    def capacity(self):
        return self.__capacity

    @property
    def occupancy(self):
        return self.__occupancy
    
    @property
    def vacancy(self):
        return self.__vacancy

    @property
    def avgn_serving(self):
        return self.__avgn_serving
    
    @property
    def avgn_occupying(self):
        return self.__avgn_occupying
    
    @property
    def util_serving(self):
        return self.__util_serving
    
    @property
    def util_occupying(self):
        return self.__util_occupying
    
    @property
    def pending_to_start(self):
        return self.__pending_to_start

    @property
    def serving(self):
        return self.__serving
    
    @property
    def pending_to_depart(self):
        return self.__pending_to_depart
    
    @abstractclassmethod
    def rqst_start(self, load):
        pass

    @abstractclassmethod
    def depart(self, load):
        pass


class Server(Sandbox, IServer):
    class Statics(IAssets):
        def __init__(self, capacity, service_time, id=None):
            super().__init__()
            self.__id = id
            self.__capacity = capacity
            self.__service_time = service_time

        @property
        def id(self):
            return self.__id

        @property
        def capacity(self):
            return self.__capacity
        
        @capacity.setter
        def capacity(self, value):
            self.__capacity = value

        @property
        def service_time(self):
            return self.__service_time
        
        @service_time.setter
        def service_time(self, value):
            self.__service_time = value

    def __init__(self, assets, seed=0, id=None):
        super().__init__(seed=seed, identifier=id)
        self.__assets = assets
        self.__hc_serving = self.add_hour_counter()
        self.__hc_pending_to_depart = self.add_hour_counter()
        self.__pending_to_start = []
        self.__serving = []
        self.__pending_to_depart = []
        self.__on_started = []
        self.__on_ready_to_depart = []

    @property
    def capacity(self):
        return self.__assets.capacity

    @property
    def occupancy(self):
        return len(self.__serving) + len(self.__pending_to_depart)

    @property
    def vacancy(self):
        return self.capacity - self.occupancy

    @property
    def avgn_serving(self):
        return self.__hc_serving.average_count

    @property
    def avgn_occupying(self):
        return self.__hc_serving.average_count + self.__hc_pending_to_depart.average_count

    @property
    def util_serving(self):
        return self.avgn_serving / self.capacity

    @property
    def util_occupying(self):
        return self.avgn_occupying / self.capacity

    @property
    def pending_to_start(self):
        return self.__pending_to_start

    @property
    def serving(self):
        return self.__serving
    
    @property
    def pending_to_depart(self):
        return self.__pending_to_depart

    def rqst_start(self, load):
        self.__pending_to_start.append(load)
        self.atmpt_start()
    
    def atmpt_start(self):
        if len(self.__pending_to_start) > 0 and self.vacancy > 0:
            load = self.__pending_to_start[0]
            self.__pending_to_start.pop(0)
            self.__serving.append(load)
            self.__hc_serving.observe_change(1, self.clock_time)
            for func in self.__on_started:
                if len(func) == 1:
                    func[0](load)
                else:
                    func[0](**func[1])
            self.schedule([self.ready_to_depart, {'load': load}], self.__assets.service_time)
    
    def ready_to_depart(self, load):
        self.__serving.remove(load)
        self.__pending_to_depart.append(load)
        self.__hc_serving.observe_change(-1, self.clock_time)
        self.__hc_pending_to_depart.observe_change(1, self.clock_time)
        for func in self.__on_ready_to_depart:
            if len(func) == 1:
                func[0](load)
            else:
                func[0](**func[1])
    
    def depart(self, load):
        if load in self.__pending_to_depart:
            self.__pending_to_depart.remove(load)
            self.__hc_pending_to_depart.observe_change(-1, self.clock_time)
            self.atmpt_start()

    @property
    def on_started(self):
        return self.__on_started

    @on_started.setter
    def on_started(self, value):
        self.__on_started.append(value)

    @property
    def on_ready_to_depart(self):
        return self.__on_ready_to_depart

    @on_ready_to_depart.setter
    def on_ready_to_depart(self, value):
        self.__on_ready_to_depart.append(value)
