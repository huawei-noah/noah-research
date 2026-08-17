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
import sys
import asyncio
import aioredis
from typing import Dict, List, Optional, Any
import nest_asyncio
nest_asyncio.apply()

if sys.platform == "win32":
    # Switch to the Windows Proactor event loop (better suited for I/O-intensive tasks)
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())

from .dbconfig import DBConfig


def build_file_tree(items):
    # Initialize the root node
    root = {
        "type": "directory",
        "name": "#ROOT",
        "children": [],
        "content": ""
    }

    for item in items:
        item_type = item["type"]
        item_name = item["name"]
        item_path = item["path"]
        item_content = item["content"]

        # Process the path: split it and filter out empty strings
        parts = [p for p in item_path.split("/") if p != ""]

        # Find the parent node of the current item
        current_node = root
        # Walk all path parts except the last one, ensuring the parent directories exist
        for part in parts[:-1] if parts else []:
            found = False
            # Check whether the directory exists among the current node's children
            for child in current_node["children"]:
                if child["type"] == "directory" and child["name"] == part:
                    current_node = child
                    found = True
                    break
            if not found:
                # Create a new directory if it does not exist
                new_dir = {
                    "type": "directory",
                    "name": part,
                    "children": [],
                    "content": ""
                }
                current_node["children"].append(new_dir)
                current_node = new_dir

        # Check whether a node with the current name already exists among the parent's children
        existing_index = -1
        for i, child in enumerate(current_node["children"]):
            if child["name"] == item_name:
                existing_index = i
                break

        if existing_index != -1:
            # A node with the same name exists; handle it
            existing_node = current_node["children"][existing_index]
            if existing_node["type"] == item_type:
                # Same type: update the content
                if item_type == "directory":
                    existing_node["content"] = item_content
                else:
                    current_node["children"][existing_index] = {
                        "type": item_type,
                        "name": item_name,
                        "content": item_content
                    }
            else:
                # Different type: replace the node
                new_node = {
                    "type": item_type,
                    "name": item_name,
                    "content": item_content
                }
                if item_type == "directory":
                    new_node["children"] = []
                current_node["children"][existing_index] = new_node
        else:
            # No node with the same name: create a new node
            new_node = {
                "type": item_type,
                "name": item_name,
                "content": item_content
            }
            if item_type == "directory":
                new_node["children"] = []
            current_node["children"].append(new_node)

    return root

class RedisClient:
    def __init__(self, config:DBConfig):
        self.redis: Optional[aioredis.Redis] = None
        self.agent_registry_key = "agent_registry"
        self.pubsub = None
        self.config = config

    async def connect(self):
        """Connect to the Redis server"""
        self.redis = aioredis.from_url(
            self.config.REDIS_URL,
            encoding="utf-8",
            decode_responses=True,
            max_connections=30,
        )
        self.pubsub = self.redis.pubsub()

    async def disconnect(self):
        """Disconnect from the Redis server"""
        if self.redis:
            await self.redis.close()

    async def register_agent(self, agent_info: Dict[str, str]):
        """
        Register agent information to Redis

        Fix: ensure the hset command uses the correct number of arguments (key, field, value)
        """
        if not self.redis:
            raise Exception("Redis connection not established")

        # Use agent_name as the field and the full info as the value
        await self.redis.hset(
            self.agent_registry_key,
            agent_info["agent_name"],  # field
            json.dumps(agent_info)  # value
        )
        print(f'Registered agent: {agent_info}')
        # Set the expiration, refreshed together with the heartbeat mechanism
        await self.redis.expire(self.agent_registry_key, 3600 * 2)  # expires in 2 hours

    async def get_all_agents(self) -> List[Dict[str, str]]:
        """Get information of all registered agents"""
        if not self.redis:
            raise Exception("Redis connection not established")

        agents_data = await self.redis.hgetall(self.agent_registry_key)
        return [json.loads(data) for data in agents_data.values()]

    async def push_stream_msg(self, user_id, session_id, message: Dict[str, Any], auto_merge: bool=True, auto_wait: bool=True):
        key = f'user:{user_id}:session:{session_id}'
        if auto_merge:
            # last_msg_in_cache = await self.redis.lrange(f"{key}:stream", 0, 0)
            # if len(last_msg_in_cache) > 0:
            #     last_msg_in_cache = json.loads(last_msg_in_cache[0])
            #     merged = self.merge_messages(last_msg_in_cache, message)
            #     if merged is not None:
            #         await self.redis.lpop(f"{key}:stream")
            #         await self.redis.rpush(f"{key}:stream", json.dumps(merged))
            #         return True
            await self.redis.rpush(f"{key}:tab_cache", json.dumps(message))
        await self.redis.rpush(f"{key}:stream", json.dumps(message))
        await asyncio.sleep(0.01)

    async def pop_stream_msg(self, user_id: str, session_id: str, auto_parse_json: bool=True, 
                             auto_push_to_cache: bool=True):
        res = None
        try:
            key = f'user:{user_id}:session:{session_id}'
            res = await self.redis.lpop(f"{key}:stream")
            if auto_parse_json:
                res = json.loads(res)
        except Exception as e:
            pass
        return res

    async def merge_messages(self, first: Dict[str, Any], second: Dict[str, Any]):
        """Merge two messages"""
        # Use different merge strategies depending on the event type
        if first.get("tab") and (first.get("tab") == second.get("tab")) and (first.get("task") == second.get("task")):
            if first['timestamp'] < second['timestamp']:
                first, second = second, first
                
            # merge first behind second
            # - dialogs
            
            return {
                'task': first.get("task"),
                'tab': first.get("tab"),
                'id': first.get("id"),
                'content': f"{first['content']}{second['content']}",
            }
        # Default merge strategy
        return None

    async def get_messages(self, user_id: str, session_id: str) -> List[Dict[str, Any]]:
        """Get session messages"""
        if not self.redis:
            raise Exception("Redis connection not established")

        key = f"user:{user_id}:session:{session_id}"
        messages = await self.redis.lrange(f"{key}:messages", 0, -1)
        return [json.loads(msg) for msg in messages]

    async def update_task_summary(self, user_id: str, session_id: str, summary: Dict[str, str]):
        """Update the task summary"""
        if not self.redis:
            raise Exception("Redis connection not established")

        key = f"user:{user_id}:session:{session_id}"
        # Fix possible hset errors; ensure the correct argument format is used
        for field, value in summary.items():
            await self.redis.hset(f"{key}:task_summary", field, value)

    async def get_task_summary(self, user_id: str, session_id: str) -> Dict[str, str]:
        """Get the task summary"""
        if not self.redis:
            raise Exception("Redis connection not established")

        key = f"user:{user_id}:session:{session_id}"
        return await self.redis.hgetall(f"{key}:task_summary")

    async def update_node_results(self, user_id: str, session_id: str, node_id, results: Dict[str, Any]):
        """Update the task summary"""
        if not self.redis:
            raise Exception("Redis connection not established")

        key = f"user:{user_id}:session:{session_id}:node:{node_id}"
        # Fix possible hset errors; ensure the correct argument format is used
        for field, value in results.items():
            if isinstance(value, str) and value == '<CLEAN>':
                _route = f"{key}:node_results"
                if field in {'stream', 'result_code'}:
                    await self.redis.delete(_route + f':{field}')
                else:
                    await self.redis.hdel(_route, field)
            elif field == 'stream':
                if isinstance(value, dict):
                    value = [value]
                for v in value:
                    await self.redis.rpush(f"{key}:node_results:stream", json.dumps(v))
            elif field == 'result_code':
                if isinstance(value, dict):
                    value = [value]
                for v in value:
                    await self.redis.rpush(f"{key}:node_results:codes", json.dumps(v))
            else:
                await self.redis.hset(f"{key}:node_results", field, value)

    async def get_node_results(self, user_id: str, session_id: str, node_id: str) -> Dict[str, str]:
        """Get the task summary"""
        if not self.redis:
            raise Exception("Redis connection not established")

        key = f"user:{user_id}:session:{session_id}:node:{node_id}"
        results = await self.redis.hgetall(f"{key}:node_results")

        results_stream = await self.redis.lrange(f"{key}:node_results:stream", 0, -1)
        results_stream = {'stream': [json.loads(s) for s in results_stream]}

        results_codes = dict()
        try:
            _codes = await self.redis.lrange(f"{key}:node_results:codes", 0, -1)
            if _codes:
                all_codes = [json.loads(s) for s in _codes]
                results_codes = {'result_code': build_file_tree(all_codes)}
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f'failed to fetch node codes. ({repr(e)})')

        return results | results_stream | results_codes

    async def publish_message(self, channel: str, message: Dict[str, Any]):
        """Publish a message to the specified channel"""
        if not self.redis:
            raise Exception("Redis connection not established")

        await self.redis.publish(channel, json.dumps(message))

    async def subscribe(self, channel: str):
        """Subscribe to the specified channel"""
        if not self.pubsub:
            raise Exception("Redis connection not established")

        await self.pubsub.subscribe(channel)

    async def get_next_message(self):
        """Get the next message"""
        if not self.pubsub:
            raise Exception("Redis connection not established")

        return await self.pubsub.get_message(ignore_subscribe_messages=True)
