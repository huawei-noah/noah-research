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
from demos.demo4.ping_pong_player import PingPongPlayer
from o2despy.sandbox import Sandbox


class PingPongGame(Sandbox):
    def __init__(self, index_player1, delay_time_expected_player1, delay_time_CV_player1,
                 index_player2, delay_time_expected_player2, delay_time_CV_player2, seed=0):
        super().__init__(seed=seed)
        self.index_player1 = index_player1
        self.delay_time_expected_player1 = delay_time_expected_player1
        self.delay_time_CV_player1 = delay_time_CV_player1
        self.index_player2 = index_player2
        self.delay_time_expected_player2 = delay_time_expected_player2
        self.delay_time_CV_player2 = delay_time_CV_player2

        self.player1 = self.add_child(
            PingPongPlayer(self.index_player1, self.delay_time_expected_player1, self.delay_time_CV_player1))
        self.player2 = self.add_child(
            PingPongPlayer(self.index_player2, self.delay_time_expected_player2, self.delay_time_CV_player2))

        # self.player1.on_send = [[self.player2.receive]]
        # self.player2.on_send = [[self.player1.receive]]
        self.player1.on_send += self.player2.receive
        self.player2.on_send += self.player1.receive

        self.schedule([self.player1.receive])


if __name__ == '__main__':
    # Demo 4
    print("Demo 4 - PingPong")
    sim4 = PingPongGame(1, 1, 1, 2, 2, 2)
    sim4.run(duration=datetime.timedelta(minutes=5))
