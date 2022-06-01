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


class IGenerator(ISandbox):
	@property
	def start_time(self):
		return self.__start_time

	@property
	def is_on(self):
		return self.__is_on

	@property
	def count(self):
		return self.__count

	@abstractclassmethod
	def start(self):
		pass

	@abstractclassmethod
	def end(self):
		pass


class Generator(Sandbox):
	def __init__(self, inter_arrival_time, seed=0, id=None):
		super().__init__(seed=seed, identifier=id)
		self.__inter_arrival_time = inter_arrival_time
		self.__start_time = None
		self.__seed = seed
		self.__id = id
		self.__is_on = False
		self.__count = 0
		self.__on_arrive = []

	@property
	def start_time(self):
		return self.__start_time

	@property
	def is_on(self):
		return self.__is_on

	@property
	def count(self):
		return self.__count

	@property
	def id(self):
		return self.__id

	@property
	def inter_arrival_time(self):
		return self.__inter_arrival_time

	@inter_arrival_time.setter
	def inter_arrival_time(self, value):
		self.__inter_arrival_time = value

	def start(self):
		if not self.__is_on:
			if self.__inter_arrival_time is None:
				raise Exception('Inter-arrival time is null')
			self.__is_on = True
			self.__start_time = self.clock_time
			self.__count = 0
			self.schedule_to_arrive()

	def end(self):
		if self.__is_on:
			self.__is_on = False

	def schedule_to_arrive(self):
		self.schedule([self.arrive], self.__inter_arrival_time)

	def arrive(self):
		if self.__is_on:
			self.__count += 1
			self.schedule_to_arrive()
			for func in self.__on_arrive:
				if len(func) == 1:
					func[0]()
				else:
					func[0](**func[1])

	@property
	def on_arrive(self):
		return self.__on_arrive

	@on_arrive.setter
	def on_arrive(self, value):
		self.__on_arrive.append(value)

	def warmup_handler(self):
		self.__count = 0

