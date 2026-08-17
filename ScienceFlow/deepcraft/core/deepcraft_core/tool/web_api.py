# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# This program is free software; you can redistribute it and/or modify it under
# the terms of the MIT license.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY
# WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A
# PARTICULAR PURPOSE. See the MIT License for more details.
#
# The name of Huawei and the contributors may not be used to endorse or promote
# products derived from this software without specific prior written permission.

import json
import uuid
import asyncio
from time import sleep, time
from subprocess import Popen
from deepcraft_core.storage.dbconfig import DBConfig
from deepcraft_core.storage.redis_client import RedisClient

def get_timestamp():
    return int(round(time() * 1000))

class WebAPI:
    def __init__(self, user_id, session_id, db_config: DBConfig=None):
        self.redis = None
        if db_config is not None:
            self.redis = RedisClient(db_config)
            
        self.user_id = user_id
        self.session_id = session_id
        
        self.connected = False
        
    async def connect2redis(self):
        if self.redis is not None and not self.connected:
            await self.redis.connect()
            self.connected = True

    @staticmethod
    def uuid():
        return str(uuid.uuid4())

    @staticmethod
    def get_result_node_id(user_id, session_id):
        return f'RESULT@{user_id}@{session_id}'

    async def push_message(self, msg: dict):
        await self.connect2redis()
        if self.redis is not None:
            msg['timestamp'] = get_timestamp()
            if 'data' in msg:
                if isinstance(msg['data'], dict):
                    msg['data']['timestamp'] = msg['timestamp']
            await self.redis.push_stream_msg(self.user_id, self.session_id, msg)

    async def update_node_results(self, node_id: str, data):
        await self.connect2redis()
        if self.redis is not None:
            await self.redis.update_node_results(self.user_id, self.session_id, node_id, data)

    @staticmethod
    def get_timestamp():
        return get_timestamp()

class StreamManager(WebAPI):
    def __init__(self, user_id, session_id, db_config: DBConfig=None, timeout=600):
        super().__init__(user_id, session_id, db_config)
        self.timeout = timeout
        self.finished = False

    async def run_until_finished(self, proc: Popen):
        await self.connect2redis()
        self.finished = self.redis is None
        t_start = time()
        _got_finished_flag = False
        while not self.finished:
            # ev = loop.run_until_complete(self.redis.pop_stream_msg(self.user_id, self.session_id))
            # _t = time()
            ev = await self.redis.pop_stream_msg(self.user_id, self.session_id)
            # print(f'> REDIS read elapsed: {time() - _t: 0.4f} sec')

            if ev is not None:
                t_start = time()
                _data = json.dumps(ev)
                yield f'data: {_data}\n\n'
                if ev['event_type'] == 'error':
                    break

                # if len(_data) > 1024:
                #     yield 'data: <CHUNK_START>\n\n'
                #     # await asyncio.sleep(0.01)
                #     sleep(0.01)
                #
                #     for i in range(0, len(_data), 1024):
                #         yield f'data: {_data[i:i + 1024]}\n\n'
                #         # await asyncio.sleep(0.01)
                #         sleep(0.01)
                #
                #     yield 'data: <CHUNK_FINISH>\n\n'
                #     # await asyncio.sleep(0.01)
                #     sleep(0.01)
                # else:
                #     yield f'data: {_data}\n\n'

                if ev['event_type'] == 'complete':
                    _got_finished_flag = True
            elif _got_finished_flag:
                break

            if time() - t_start > self.timeout:
                print('timed out')
                self.finished = True

            # await asyncio.sleep(0.01)
            sleep(0.01)

            if proc.poll() is not None:
                print('task terminated on its own')
                self.finished = True
                break

        print('<current flow has ended>')
        yield "data: {}\n\n".format(json.dumps({
            'event_type': 'complete',
            'data': 'all tasks completed',
            'timestamp': get_timestamp()
        }))
