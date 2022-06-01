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


from abc import ABC, abstractmethod
import datetime


class IReadOnlyHourCounter(ABC):
    @abstractmethod
    def last_time(self):
        pass

    @abstractmethod
    def last_count(self):
        pass

    @abstractmethod
    def paused(self):
        pass

    @abstractmethod
    def total_increment(self):
        pass

    @abstractmethod
    def total_decrement(self):
        pass

    @abstractmethod
    def increment_rate(self):
        pass

    @abstractmethod
    def decrement_rate(self):
        pass

    @abstractmethod
    def total_hours(self):
        pass

    @abstractmethod
    def working_time_ratio(self):
        pass

    @abstractmethod
    def cum_value(self):
        pass

    @abstractmethod
    def average_count(self):
        pass

    @abstractmethod
    def average_duration(self):
        pass

    @abstractmethod
    def log_file(self):
        pass
    

class IHourCounter(IReadOnlyHourCounter, ABC):
    @abstractmethod
    def observe_count(self, count, clock_time):
        pass
    
    @abstractmethod
    def observe_change(self, count, clock_time):
        pass
    
    @abstractmethod
    def pause(self, *args):
        pass

    @abstractmethod
    def resume(self, clock_time):
        pass


class ReadOnlyHourCounter(IReadOnlyHourCounter):
    def __init__(self, hour_counter):
        self.__hour_counter = hour_counter

    @property
    def last_time(self):
        return self.__hour_counter.last_time

    @property
    def last_count(self):
        return self.__hour_counter.last_count
        
    @property
    def paused(self):
        return self.__hour_counter.paused

    @property
    def total_increment(self):
        return self.__hour_counter.total_increment

    @property
    def increment_rate(self):
        return self.__hour_counter.increment_rate

    @property
    def decrement_rate(self):
        return self.__hour_counter.decrement_rate

    @property
    def total_decrement(self):
        return self.__hour_counter.total_decrement

    @property
    def total_hours(self):
        return self.__hour_counter.total_hours

    @property
    def working_time_ratio(self):
        return self.__hour_counter.working_time_ratio

    @property
    def cum_value(self):
        return self.__hour_counter.cum_value

    @property
    def average_count(self):
        return self.__hour_counter.average_count

    @property
    def average_duration(self):
        return self.__hour_counter.average_duration

    @property
    def log_file(self):
        return self.__hour_counter.log_file
    
    @log_file.setter
    def log_file(self, value):
        self.__hour_counter.log_file = value


class HourCounter(IHourCounter):
    def __init__(self, sandbox, keep_history=False, initial_time=None):
        self.__sandbox = sandbox
        self.__initial_time = initial_time if initial_time else datetime.datetime.min
        self.__last_time = self.__initial_time
        self.__last_count = 0
        self.__total_increment = 0
        self.__total_decrement = 0
        self.__total_hours = 0
        self.__cum_value = 0
        self.__keep_history = keep_history
        self.__paused = False
        self.__log_file = None
        self.hours_for_count = {}
        self.__read_only = None
        if keep_history:
            self.__history = {}

    @property
    def last_time(self):
        return self.__last_time

    @property
    def last_count(self):
        return self.__last_count

    @property
    def total_increment(self):
        return self.__total_increment

    @property
    def total_decrement(self):
        return self.__total_decrement

    @property
    def total_hours(self):
        return self.__total_hours

    def update_to_clock_time(self):
        if self.__last_time != self.__sandbox.clock_time:
            self.observe_count(self.__last_count)

    @property
    def working_time_ratio(self):
        self.update_to_clock_time()
        if self.__last_time == self.__initial_time:
            return 0
        return self.__total_hours / (self.__last_time - self.__initial_time).hour()
    
    @property
    def cum_value(self):
        return self.__cum_value

    @property
    def average_count(self):
        self.update_to_clock_time()
        if self.__total_hours == 0:
            return self.__last_count
        return self.__cum_value / self.__total_hours
    
    @property
    def average_duration(self):
        self.update_to_clock_time()
        hours = self.average_count / self.decrement_rate
        if not hours:
            hours = 0
        return hours / 3600
    
    @property
    def keep_history(self):
        return self.__keep_history

    @property
    def paused(self):
        return self.__paused        

    @property
    def history(self):
        if not self.__keep_history:
            return None
        return [((key - self.__initial_time) / 3600, self.__history[key])
                for key in sorted(self.__history.keys(), key=lambda x: x)]

    @property
    def increment_rate(self):
        self.update_to_clock_time()
        return self.__total_increment / self.__total_hours

    @property
    def decrement_rate(self):
        self.update_to_clock_time()
        return self.__total_decrement / self.__total_hours

    @property
    def log_file(self):
        return self.__log_file
    
    @log_file.setter
    def log_file(self, value):
        self.__log_file = value
        if self.__log_file:
            with open(self.__log_file, 'w') as f:
                f.writelines('Hours ,Count ,Remark')
                f.writelines(f'{self.__total_hours}, {self.__last_count}')

    def observe_count(self, count, clock_time=None):
        if clock_time is None:
            clock_time = self.__sandbox.clock_time
        elif clock_time is not None:
            if clock_time != self.__sandbox.clock_time:
                raise Exception("Clock-time is not consistent with the Sandbox.")
        if clock_time < self.__last_time:
            raise Exception('Time of new count cannot be earlier than current time.')
        if not self.__paused:
            hours = (clock_time - self.__last_time).total_seconds() / float(3600)
            self.__total_hours += hours
            self.__cum_value += hours * self.__last_count
            if count > self.__last_count:
                self.__total_increment += count - self.__last_count
            else:
                self.__total_decrement += self.__last_count - count
            if self.__last_count not in self.hours_for_count.keys():
                self.hours_for_count[self.__last_count] = 0
            self.hours_for_count[self.__last_count] += hours
        if self.__log_file:
            with open(self.__log_file, 'w') as f:
                f.write(f'{self.__total_hours}, {self.__last_count}')
                if self.__paused:
                    f.write(', Paused')
                f.writelines('')
                if count != self.__last_count:
                    f.write(f'{self.__total_hours}, {self.__last_count}')
                    if self.__paused:
                        f.write(', Paused')
                    f.writelines('')
        self.__last_count = clock_time
        self.__last_count = count
        if self.__keep_history:
            self.__history[clock_time] = count

    def observe_change(self, change, clock_time=None):
        if clock_time is not None:
            if clock_time != self.__sandbox.clock_time:
                raise Exception("Clock-time is not consistent with the Sandbox")
        return self.observe_count(self.__last_count + change, clock_time)

    def pause(self, clock_time=None):
        if clock_time is not None:
            if clock_time != self.__sandbox.clock_time:
                raise Exception("Clock-time is not consistent with the Sandbox.")
        clock_time = self.__sandbox.clock_time
        if self.__paused:
            return
        self.observe_count(self.__last_count, clock_time)
        self.__paused = True
        if not self.__log_file:
            with open(self.__log_file, 'w') as f:
                f.writelines(f'{self.__total_hours}, {self.__last_count}, Paused')

    def resume(self, clock_time):
        if clock_time is not None:
            if clock_time != self.__sandbox.clock_time:
                raise Exception("Clock-time is not consistent with the Sandbox.")
        if not self.__paused:
            return
        self.__last_time = self.__sandbox.clock_time
        self.__paused = False
        if self.__log_file:
            with open(self.__log_file, 'w') as f:
                f.writelines(f'{self.__total_hours}, {self.__last_count}, Paused')
                f.writelines(f'{self.__total_hours}, {self.__last_count}')
         
    def warmed_up(self):
        self.__initial_time = self.__sandbox.clock_time
        self.__last_time = self.__sandbox.clock_time
        self.__total_increment = 0
        self.__total_decrement = 0
        self.__total_hours = 0
        self.__cum_value = 0
        self.hours_for_count = {}
    
    def __sort_hours_for_count(self):
        self.hours_for_count = {key: self.hours_for_count[key]
                                for key in sorted(self.hours_for_count.keys(), key=lambda x: x)}

    def percentile(self, ratio):
        self.__sort_hours_for_count()
        threshold = sum(self.hours_for_count.values()) * ratio / 100
        for key, value in self.hours_for_count.items():
            threshold -= value
            if threshold <= 0:
                return key
        return float('inf')

    def histogram(self, count_interval):
        self.__sort_hours_for_count()
        __histogram = {}
        if len(self.hours_for_count):
            count_lb = 0
            cum_hours = 0
            last_key = list(self.hours_for_count.keys())[-1]
            for key, value in self.hours_for_count.items():
                if key > count_lb + count_interval or key == last_key:
                    if cum_hours > 0:
                        __histogram[count_lb] = [cum_hours, 0, 0]
                    count_lb += count_interval
                    cum_hours = value
                else:
                    cum_hours += value
        his_sum = sum(list(__histogram.values())[0])
        cum = 0
        for value in __histogram.values():
            cum += value[0]
            value[1] = value[0]
            value[2] = cum / his_sum
        return __histogram
