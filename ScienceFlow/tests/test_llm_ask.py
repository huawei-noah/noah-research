#!/usr/bin/env python3
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

"""用当前 YAML + 环境变量（与 ScienceFlow CLI 相同）做一次 ``llm.ask`` 连通性验证。

在仓库根目录执行::

    python tests/test_llm_ask.py
    python tests/test_llm_ask.py --config path/to.yaml --stage code
    python tests/test_llm_ask.py -p "用一句话回答：1+1等于几？"

依赖：已安装 ScienceFlow / deepcraft，且 ``API_KEY``+``BASE_URL`` 或 YAML 中已配置对应 stage。
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path
from pprint import pformat

from deepcraft_core import Message

from scienceflow.config.settings import Config, StageConfig, load_cfg
from scienceflow.core.key_pool import parse_key_env
from scienceflow.core.llm_http import aclose_llm_clients, llm_extra_client_kwargs
from scienceflow.core.agent_runtime import _build_llm


def _bootstrap_env() -> None:
    try:
        from dotenv import load_dotenv
    except ImportError:
        return
    load_dotenv()


def _stage(cfg: Config, name: str) -> StageConfig:
    if name not in ("code", "feedback"):
        raise SystemExit(f"未知 stage: {name!r}，应为 code|feedback")
    return getattr(cfg.agent, name)


def _stage_ready(stage: StageConfig) -> tuple[bool, str]:
    if stage.api_keys:
        pass
    elif not (stage.api_key or "").strip():
        return False, "缺少密钥：请设置 API_KEY / API_KEYS 或在 YAML 中配置 agent.<stage>.api_key(s)"
    if stage.base_urls:
        pass
    elif not (stage.base_url or "").strip():
        return False, "缺少 Base URL：请设置 BASE_URL / BASE_URLS 或在 YAML 中配置 agent.<stage>.base_url(s)"
    return True, ""


def _summarize_stage(stage: StageConfig) -> str:
    keys_n = len(stage.api_keys) if stage.api_keys else (1 if stage.api_key else 0)
    urls_n = len(stage.base_urls) if stage.base_urls else (1 if stage.base_url else 0)
    url_preview = (stage.base_urls[0] if stage.base_urls else stage.base_url)[:48]
    return (
        f"model={stage.model!r}, endpoints≈{urls_n} url(s) ({url_preview!r}…), "
        f"keys={keys_n}, max_tokens={stage.max_tokens}, use_proxy={stage.use_proxy}"
    )


def _shared_llm_kwargs(stage: StageConfig) -> dict[str, object]:
    """与 ``scienceflow.core.agent_runtime._build_llm`` 中传入 OnlineLLM 的共享参数一致。"""
    return dict(
        model=stage.model,
        max_tokens=stage.max_tokens,
        stream=True,
        tracker=True,
        **llm_extra_client_kwargs(stage),
    )


def _mask_api_key(key: str | None) -> str:
    if not (key or "").strip():
        return "(empty)"
    s = key.strip()
    if len(s) <= 8:
        return "***"
    return f"{s[:4]}…{s[-4:]} (len={len(s)})"


def _display_value(key: str, val: object) -> object:
    if key == "api_key":
        return _mask_api_key(val if isinstance(val, str) else None)
    if key == "http_asyncclient" and val is not None:
        cls = type(val).__name__
        return f"<{cls} id={id(val):#x}>"
    if key == "headers" and isinstance(val, dict):
        out: dict[str, str] = {}
        for hk, hv in val.items():
            lk = hk.lower()
            if any(x in lk for x in ("auth", "token", "key", "secret")):
                out[hk] = _mask_api_key(str(hv))
            else:
                out[hk] = str(hv)
        return out
    return val


def _print_llm_construct_args(stage: StageConfig) -> None:
    """打印即将用于构造 ``OnlineLLM`` / ``PooledLLM`` 的入参（与 ``_build_llm`` 对齐，密钥脱敏）。"""
    shared = _shared_llm_kwargs(stage)
    shared_lines = {
        k: _display_value(k, v) for k, v in sorted(shared.items(), key=lambda x: x[0])
    }
    print("[test_llm_ask] 等价构造调用（与 operators._build_llm 一致，api_key 已脱敏）:")
    if stage.api_keys:
        endpoints = parse_key_env(
            api_keys_csv=",".join(stage.api_keys),
            base_urls_csv=",".join(stage.base_urls) if stage.base_urls else None,
            fallback_url=stage.base_url,
        )
        ep_display = [(url, _mask_api_key(k)) for url, k in endpoints]
        print("  PooledLLM.from_endpoints(")
        print(f"      endpoints={pformat(ep_display)},")
        print("      # 以下 **kwargs 与 operators._build_llm 中传入每个 OnlineLLM 的相同：")
        for line in pformat(shared_lines, width=96).splitlines():
            print(f"      {line}")
        print("  )")
    else:
        one: dict[str, object] = {
            "base_url": stage.base_url,
            "api_key": _mask_api_key(stage.api_key),
            **shared_lines,
        }
        print("  OnlineLLM(")
        for line in pformat(one, width=96).splitlines():
            print(f"      {line}")
        print("  )")


async def _run_ask(stage_name: str, cfg: Config, prompt: str, *, stream: bool) -> None:
    stage = _stage(cfg, stage_name)
    ok, err = _stage_ready(stage)
    if not ok:
        print(err, file=sys.stderr)
        raise SystemExit(2)

    print(f"[test_llm_ask] stage={stage_name} {_summarize_stage(stage)}")
    _print_llm_construct_args(stage)
    llm = _build_llm(stage)
    try:
        text = await llm.ask(
            messages=[Message.user_message(prompt)],
            system_msgs=[Message.system_message("You are a helpful assistant. Be concise.")],
            stream=stream,
        )
    finally:
        await aclose_llm_clients(llm)

    print("--- reply ---")
    print(text.strip() if text else "(empty)")
    print("-------------")

    for attr in ("_last_call_input_tokens", "_last_call_output_tokens", "_last_call_ttft"):
        if hasattr(llm, attr):
            print(f"{attr}: {getattr(llm, attr)}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify OpenAI-compatible LLM via ScienceFlow config.")
    parser.add_argument(
        "--config",
        "-c",
        default=None,
        help="YAML 路径；默认 scienceflow/config/default.yaml",
    )
    parser.add_argument(
        "--stage",
        "-s",
        default="code",
        choices=("code", "feedback"),
        help="使用 agent 下哪一阶段的 LLM 配置（默认 code）",
    )
    parser.add_argument(
        "--prompt",
        "-p",
        default="Reply with one short English sentence: what is 2+2?",
        help="发给模型的用户消息",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="使用流式请求（默认关闭，便于看整段回复）",
    )
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="打印 DEBUG 日志",
    )
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    _bootstrap_env()
    cfg_path = Path(args.config) if args.config else None
    cfg = load_cfg(cfg_path)

    try:
        asyncio.run(_run_ask(args.stage, cfg, args.prompt, stream=args.stream))
    except SystemExit as e:
        return int(e.code) if isinstance(e.code, int) else 1
    except Exception as e:
        print(f"[test_llm_ask] 失败: {e}", file=sys.stderr)
        if args.verbose:
            logging.getLogger("test_llm_ask").exception("detail")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
