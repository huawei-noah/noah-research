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


import random
from abc import ABC, abstractmethod
from sortedcontainers import SortedSet
from o2despy.event import Event
from o2despy.hour_counter import HourCounter
from o2despy.assets import IAssets
import time
import datetime


class ISandbox(ABC):
    def __init__(self, index=None, identifier=None, parent=None, children=None,
                 clock_time=None, log_file=None, debug_mode=None):
        self.__index = index
        self.__id = identifier
        self.__parent = parent
        self.__children = children
        self.__clock_time = clock_time
        self.__log_file = log_file
        self.__debug_mode = debug_mode

    @property
    def index(self):
        return self.__index

    @property
    def id(self):
        return self.__id

    @property
    def parent(self):
        return self.__parent

    @property
    def children(self):
        return self.__children

    @property
    def clock_time(self):
        return self.__clock_time

    @property
    def log_file(self):
        return self.__log_file

    @log_file.setter
    def log_file(self, value):
        self.__log_file = value

    @property
    def debug_mode(self):
        return self.__debug_mode

    @debug_mode.setter
    def debug_mode(self, value):
        self.__debug_mode = value

    @abstractmethod
    def run(self, *args, **kwargs):
        pass

    @abstractmethod
    def warmup(self, *args, **kwargs):
        pass


class SandboxStatics(IAssets):
    def __init__(self):
        super().__init__()


class Sandbox(ISandbox):
    __count = 0

    def __init__(self, seed=0, identifier=None):
        super().__init__(identifier)
        assets = SandboxStatics()
        self.__assets = assets
        self.__id = identifier
        Sandbox.__count += 1
        self.__index = Sandbox.__count
        self.__future_event_list = SortedSet()
        self.__head_event = None
        self.__parent = None
        self.__children = []
        self.__seed = self._initialise_seed(seed)
        self.__clock_time = datetime.datetime.min
        self.__realtime_for_last_run = None
        self.__on_warmed_up = []
        self.__hour_counters = []

    @property
    def assets(self):
        return self.__assets

    @assets.setter
    def assets(self, value):
        self.__assets = value

    @property
    def id(self):
        return self.__id

    @property
    def index(self):
        return self.__index

    @property
    def future_event_list(self):
        return self.__future_event_list

    @property
    def parent(self):
        return self.__parent

    @parent.setter
    def parent(self, value):
        self.__parent = value

    @property
    def children(self):
        return self.__children

    @property
    def seed(self):
        return self.__seed

    @seed.setter
    def seed(self, value):
        self.__seed = value
        random.seed(value)

    @property
    def clock_time(self):
        if self.__parent is None:
            return self.__clock_time
        return self.__parent.clock_time

    @property
    def on_warmed_up(self):
        return self.__on_warmed_up

    @on_warmed_up.setter
    def on_warmed_up(self, value):
        self.__on_warmed_up.append(value)

    @property
    def hour_counters(self):
        return self.__hour_counters

    @property
    def log_file(self):
        return self.__log_file

    @log_file.setter
    def log_file(self, log_file):
        self.__log_file = log_file

    def _initialise_seed(self, seed):
        """
        Initialise seed. If seed is None and parent is None, set default seed value to be 0

        :param seed:
        :return:
        """
        if seed is not None:
            random.seed(seed)
        elif seed is None and self.__parent is None:
            seed = 0
            random.seed(seed)

        return seed

    def schedule(self, action, clock_time=None, tag=None):
        if len(action) > 1:
            paras = list(action[1])[0] if isinstance(action[1], list) else action[1]

        if clock_time is None:
            clock_time = datetime.timedelta(seconds=0)
            self.__future_event_list.add(Event(owner=self, action=action, scheduled_time=self.clock_time + clock_time, tag=tag))
        elif type(clock_time) is datetime.datetime:
            self.__future_event_list.add(Event(owner=self, action=action, scheduled_time=clock_time, tag=tag))
        elif type(clock_time) is datetime.timedelta:
            self.__future_event_list.add(
                Event(owner=self, action=action, scheduled_time=self.clock_time + clock_time, tag=tag))
        else:
            raise TypeError()

    @property
    def head_event(self):
        head_event = None
        if len(self.__future_event_list) > 0:
            head_event = self.__future_event_list[0]
        for child in self.__children:
            child_head_event = child.head_event
            if head_event is None or (child_head_event is not None and child_head_event < head_event):
                head_event = child_head_event
        return head_event

    def get_parent_seed_value(self):
        """
        From the parent Sandbox, ge
        :return:
        """
        if self.__parent is not None:
            self.__parent.get_parent_seed_value()
        else:
            return random.randint(1, 10000)

    def run(self, *args, **kwargs):
        if kwargs == {}:
            if self.__parent is not None:
                return self.__parent.run()
            head = self.head_event
            if head is None:
                return False
            head.owner.future_event_list.discard(head)
            self.__clock_time = head.scheduled_time
            head.invoke()
            return True
        elif 'duration' in kwargs:
            if self.__parent is not None:
                return self.__parent.run(duration=kwargs['duration'])
            return self.run(terminate=self.clock_time + kwargs['duration'])
        elif 'terminate' in kwargs:
            if self.__parent is not None:
                return self.__parent.run(terminate=kwargs['terminate'])
            n = 0
            step_time = time.time()
            while True:
                n += 1
                head = self.head_event
                if head is not None and head.scheduled_time <= kwargs['terminate']:  # Finish all event or time out
                    start_time = time.time()
                    self.run()
                    use_time = time.time() - start_time
                    use_time = 'Time_out_:{}'.format(use_time) if use_time > 0.02 else use_time
                    if n % 200 == 0:
                        current_time = time.time()
                        step_time = current_time

                else:
                    self.__clock_time = kwargs['terminate']
                    return head is not None
        elif 'event_count' in kwargs:
            if self.__parent is not None:
                return self.__parent.run(event_count=kwargs['event_count'])
            while kwargs['event_count'] > 0:
                kwargs['event_count'] -= 1
                r = self.run()
                if not r:
                    return False
            return True
        elif 'speed' in kwargs:
            if self.__parent is not None:
                return self.__parent.run(speed=kwargs['speed'])
            rtn = True
            if self.__realtime_for_last_run is not None:
                time_gap = datetime.datetime.now() - self.__realtime_for_last_run
                time_gap = datetime.timedelta(seconds=time_gap.total_seconds() * kwargs['speed'])
                rtn = self.run(terminate=self.clock_time + time_gap)
            self.__realtime_for_last_run = datetime.datetime.now()
            return rtn
        else:
            raise TypeError()

    def add_child(self, child):
        """
        Add a child Sandbox under the current parent Sandbox, set the seed of the child if None from the seed value of
        the parent Sandbox, then add on_warmed_up.

        :param child: child Sandbox
        :return: child Sandbox
        """
        self.__children.append(child)
        child.parent = self
        if child.seed is None:
            child.seed = self.get_parent_seed_value()
        self.__on_warmed_up += child.on_warmed_up
        return child

    def add_hour_counter(self, keep_history=False):
        hc = HourCounter(self, keep_history=keep_history)
        self.__hour_counters.append(hc)
        self.__on_warmed_up.append([hc.warmed_up])
        return hc

    def to_string(self):
        _id = self.__id
        if self.__id is None or len(self.__id) == 0:
            _id = type(self)
        _id += '#' + str(self.__index)
        return _id

    def warmup(self, *args, **kwargs):
        if 'period' in kwargs:
            if self.__parent is not None:
                return self.__parent.warmup(kwargs['period'])
            return self.warmup(till=self.clock_time + kwargs['period'])
        elif 'till' in kwargs:
            if self.__parent is not None:
                return self.__parent.warmup(kwargs['till'])
            result = self.run(terminate=kwargs['till'])
            self._invoke_warmup()
            return result

    def _invoke_warmup(self):
        """
        Invoke the function(s) in on_warmed_up after warmup is completed.
        """
        for func in self.__on_warmed_up:
            if len(func) == 1:
                func[0]()
            else:
                func[0](**func[1])

    def warmup_handler(self):
        """Currently has no usage. This is for adding actions to be triggered when warmup started"""
        return -1
