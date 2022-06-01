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


class BirthDeath(Sandbox):
    def __init__(self, hourly_birth_rate, hourly_death_rate, seed=0):
        super().__init__(seed=seed)
        self.hourly_birth_rate = hourly_birth_rate
        self.hourly_death_rate = hourly_death_rate
        self.population = self.add_hour_counter()

        # self.schedule([self.birth], timedelta(seconds=0))
        self.schedule([self.birth])

    def birth(self):
        self.population.observe_change(1)
        print("{0}\tBirth (Population: #{1}!)".format(self.clock_time, self.population.last_count))
        self.schedule([self.birth], timedelta(hours=round(random.expovariate(self.hourly_birth_rate), 2)))
        self.schedule([self.death], timedelta(hours=round(random.expovariate(self.hourly_death_rate), 2)))

    def death(self):
        self.population.observe_change(-1)
        print("{0}\tDeath (Population: #{1}!)".format(self.clock_time, self.population.last_count))


if __name__ == '__main__':
    # Demo 2
    print("Demo 2 - Birth Death Process")
    sim = BirthDeath(20, 1, seed=1)
    sim.warmup(period=datetime.timedelta(hours=24))
    sim.run(duration=datetime.timedelta(hours=30))
