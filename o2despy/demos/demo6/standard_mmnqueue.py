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
import math
from o2despy.sandbox import Sandbox
from standard import generator, queue, server, load


class mmn_queue_modular(Sandbox):
    def __init__(self, arrival_rate, service_rate, capacity):
        super().__init__()
        self.__generator = self.add_child(generator.Generator(datetime.timedelta(seconds=arrival_rate)))
        self.__queue = self.add_child(queue.Queue(math.inf))
        self.__server = self.add_child(
            server.Server(server.Server.Statics(capacity, datetime.timedelta(seconds=service_rate))))

        self.__generator.on_arrive = [self.__queue.rqst_enqueue, {'load': load.Load()}]
        self.__generator.on_arrive = [self.arrive]

        self.__queue.on_enqueued = [self.__server.rqst_start]
        self.__server.on_started = [self.__queue.dequeue]

        self.__server.on_ready_to_depart = [self.__server.depart]
        self.__server.on_ready_to_depart = [self.depart]

        self.__generator.start()

    def arrive(self):
        print('arrive at ', self.clock_time)

    def depart(self, load):
        print('depart at ', self.clock_time)


if __name__ == '__main__':
    # Demo 6
    print("Demo 6 - MMNQueue using standard library")
    sim = mmn_queue_modular(60, 80, 1)
    sim.run(event_count=10)


