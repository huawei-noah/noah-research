"""OrderedKeyedDispatcher 单元测试：分组串行、跨组并发、全局并发上限、背压。"""

from __future__ import annotations

import asyncio

import pytest
from mindmemos.infra.kafka.dispatcher import OrderedKeyedDispatcher


@pytest.mark.asyncio
async def test_same_key_strictly_serial():
    """同一 key 严格按提交顺序串行处理。"""
    order: list[int] = []
    active = 0
    max_active = 0

    async def process(item):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.01)
        order.append(item)
        active -= 1

    async def on_complete(item):
        pass

    d = OrderedKeyedDispatcher(
        global_max_concurrency=8,
        per_key_max_concurrency=1,
        max_buffered=100,
        process=process,
        on_complete=on_complete,
    )
    for i in range(5):
        await d.submit("user-a", i)
    await d.drain()

    assert order == [0, 1, 2, 3, 4]  # 顺序保持
    assert max_active == 1  # 同 key 任意时刻只有一个在跑


@pytest.mark.asyncio
async def test_different_keys_run_concurrently():
    """不同 key 之间可并发处理。"""
    started = asyncio.Event()
    both_running = asyncio.Event()
    running = 0

    async def process(item):
        nonlocal running
        running += 1
        if running >= 2:
            both_running.set()
        started.set()
        await asyncio.sleep(0.05)
        running -= 1

    d = OrderedKeyedDispatcher(
        global_max_concurrency=8,
        per_key_max_concurrency=1,
        max_buffered=100,
        process=process,
        on_complete=lambda _i: asyncio.sleep(0),
    )
    await d.submit("user-a", 1)
    await d.submit("user-b", 2)
    await asyncio.wait_for(both_running.wait(), timeout=1.0)
    await d.drain()


@pytest.mark.asyncio
async def test_global_concurrency_cap():
    """全局并发不超过 global_max_concurrency，即便有更多 key。"""
    active = 0
    max_active = 0

    async def process(item):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1

    d = OrderedKeyedDispatcher(
        global_max_concurrency=2,
        per_key_max_concurrency=1,
        max_buffered=100,
        process=process,
        on_complete=lambda _i: asyncio.sleep(0),
    )
    for i in range(6):
        await d.submit(f"user-{i}", i)
    await d.drain()

    assert max_active <= 2


@pytest.mark.asyncio
async def test_shared_global_concurrency_cap_across_dispatchers():
    """共享 semaphore 可限制多个 consumer/dispatcher 的合计并发。"""
    active = 0
    max_active = 0
    shared = asyncio.Semaphore(2)

    async def process(item):
        nonlocal active, max_active
        active += 1
        max_active = max(max_active, active)
        await asyncio.sleep(0.02)
        active -= 1

    dispatchers = [
        OrderedKeyedDispatcher(
            global_max_concurrency=8,
            per_key_max_concurrency=1,
            max_buffered=100,
            process=process,
            on_complete=lambda _i: asyncio.sleep(0),
            shared_global_semaphore=shared,
        )
        for _ in range(2)
    ]

    for index, dispatcher in enumerate(dispatchers):
        for item in range(4):
            await dispatcher.submit(f"{index}:{item}", item)
    await asyncio.gather(*(dispatcher.drain() for dispatcher in dispatchers))

    assert max_active <= 2


@pytest.mark.asyncio
async def test_non_positive_concurrency_config_is_rejected():
    """并发和缓冲上限必须显式配置为正整数，避免静默掩盖部署错误。"""
    with pytest.raises(ValueError, match="global_max_concurrency"):
        OrderedKeyedDispatcher(
            global_max_concurrency=0,
            per_key_max_concurrency=1,
            max_buffered=10,
            process=lambda _i: asyncio.sleep(0),
            on_complete=lambda _i: asyncio.sleep(0),
        )

    with pytest.raises(ValueError, match="per_key_max_concurrency"):
        OrderedKeyedDispatcher(
            global_max_concurrency=1,
            per_key_max_concurrency=0,
            max_buffered=10,
            process=lambda _i: asyncio.sleep(0),
            on_complete=lambda _i: asyncio.sleep(0),
        )

    with pytest.raises(ValueError, match="max_buffered"):
        OrderedKeyedDispatcher(
            global_max_concurrency=1,
            per_key_max_concurrency=1,
            max_buffered=0,
            process=lambda _i: asyncio.sleep(0),
            on_complete=lambda _i: asyncio.sleep(0),
        )


@pytest.mark.asyncio
async def test_backpressure_blocks_submit_without_dropping():
    """缓冲满时 submit 阻塞，所有消息最终都被处理，无丢失。"""
    completed: list[int] = []
    release = asyncio.Event()

    async def process(item):
        await release.wait()

    async def on_complete(item):
        completed.append(item)

    d = OrderedKeyedDispatcher(
        global_max_concurrency=1,
        per_key_max_concurrency=1,
        max_buffered=2,
        process=process,
        on_complete=on_complete,
    )
    # 填满缓冲（2 条），第三条 submit 应阻塞。
    await d.submit("k", 0)
    await d.submit("k", 1)
    blocked = asyncio.create_task(d.submit("k", 2))
    await asyncio.sleep(0.02)
    assert not blocked.done()  # 背压生效

    release.set()
    await blocked
    await d.drain()

    assert sorted(completed) == [0, 1, 2]  # 一条都不丢
