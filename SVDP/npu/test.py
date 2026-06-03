# Copyright (C) 2026. Huawei Technologies Co., Ltd. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"),
# to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense,
# and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE
# WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR
# COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

from __future__ import annotations

import torch
import torch_npu
from kernel import FeedForward, SparseFeedForward
from torch_npu import profiler
from torch_npu.profiler import AiCMetrics

WARMUP_ITERATIONS = 30
PROFILE_ITERATIONS = 1000
SPARSITY_LEVELS = [20, 50, 80, 95]
MODEL_NAMES = ["LLAMA2", "QWEN2", "MISTRAL"]


def verify_result(output: torch.Tensor, golden: torch.Tensor, err: float) -> bool:
    output = output.flatten()
    golden = golden.flatten()

    close = torch.isclose(output, golden, rtol=err, atol=err, equal_nan=True)
    diff_indexes = (~close).nonzero(as_tuple=True)[0]
    print(f"diff len: {len(diff_indexes)}")

    for i, idx in enumerate(diff_indexes):
        golden_val = golden[idx].item()
        output_val = output[idx].item()
        rdiff = abs(output_val - golden_val) / (golden_val + 1e-10)
        print(f"data index: {idx:06d}, expected: {golden_val:-.9f}, actual: {output_val:-.9f}, rdiff: {rdiff:-.6f}")
        if i == 64:
            break

    error_ratio = len(diff_indexes) / len(golden)
    print(f"error ratio: {error_ratio:.4f}, tolerance: {err:.4f}")
    return error_ratio <= err


def bench(fn, steps: int = 10) -> None:
    start_ev = torch.npu.Event(enable_timing=True)
    end_ev = torch.npu.Event(enable_timing=True)

    for _ in range(WARMUP_ITERATIONS):
        fn()

    torch.npu.synchronize()

    time_list = []
    for _ in range(steps):
        start_ev.record()
        fn()
        end_ev.record()

        torch.npu.synchronize()
        cur_time = start_ev.elapsed_time(end_ev)
        time_list.append(cur_time)

    mean_time = sum(time_list) / len(time_list)
    print(f"Mean time: {mean_time:.4f} ms")


def bench_profiler(fn, steps: int, profiler_instance) -> None:
    for _ in range(steps):
        fn()
        profiler_instance.step()
        torch_npu.npu.synchronize()


def gen_sparse_mask(size: int, sparsity: int) -> torch.Tensor:
    sparse_input = torch.ones(size, dtype=torch.float32)
    sparse_input[torch.randperm(size)[: int(size * sparsity / 100)]] = -1
    sparse_input = torch.nn.functional.relu(sparse_input)
    print("Sparsity level", (sparse_input == 0.0).to(torch.float32).mean().item())
    return sparse_input


def gen_idx_sparse_mask(sparse_input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    idx = torch.nonzero(sparse_input != 0.0).flatten()
    return idx, (sparse_input != 0.0).flatten()


class ModelConfig:
    _CONFIGS = {
        "LLAMA2": (11008, 4096),
        "QWEN2": (18944, 3584),
        "MISTRAL": (14336, 4096),
    }

    def __init__(self, model_type: str):
        if model_type not in self._CONFIGS:
            raise ValueError(f"Unsupported model: {model_type}")
        self.nrows, self.ncols = self._CONFIGS[model_type]
        self.name = model_type


def _create_profiler(
    name: str,
    warmup_iterations: int,
    profile_iterations: int,
    metrics: str = AiCMetrics.PipeUtilization,
) -> profiler.profile:
    """Create and configure the NPU profiler."""
    profiler_config = torch_npu.profiler._ExperimentalConfig(
        aic_metrics=metrics,
        profiler_level=torch_npu.profiler.ProfilerLevel.Level1,
        l2_cache=False,
    )

    return profiler.profile(
        activities=[profiler.ProfilerActivity.NPU],
        schedule=profiler.schedule(
            wait=0,
            warmup=warmup_iterations,
            active=profile_iterations,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=profiler.tensorboard_trace_handler(f"perf_{name}"),
        profile_memory=True,
        experimental_config=profiler_config,
    )


def benchmark_ffn_profiler(sparsity: int, model_config: ModelConfig) -> None:
    print(f"\nTesting {model_config.name} with {sparsity}% sparsity")
    hidden_dim, dim = model_config.nrows, model_config.ncols
    print(f"\nHidden dim {hidden_dim}, dim {dim}")

    total_iterations = WARMUP_ITERATIONS + PROFILE_ITERATIONS
    x = torch.rand((dim, 1)).to(torch.float16).npu()
    sparse_input = gen_sparse_mask(hidden_dim, sparsity)
    idx, mask = gen_idx_sparse_mask(sparse_input)
    idx = idx.npu()

    ff = FeedForward(dim, hidden_dim)
    sff = SparseFeedForward(ff)
    ff.zero_weights(mask)

    def run_dense():
        return ff(x)

    with _create_profiler(
        f"dense_{model_config.name}_{sparsity}",
        WARMUP_ITERATIONS,
        PROFILE_ITERATIONS,
        AiCMetrics.Memory,
    ) as profiler_instance:
        print("ffn")
        bench_profiler(run_dense, total_iterations, profiler_instance)
        y = run_dense()

    def run_sparse():
        return sff(x, idx)

    with _create_profiler(
        f"sparse_{model_config.name}_{sparsity}",
        WARMUP_ITERATIONS,
        PROFILE_ITERATIONS,
        AiCMetrics.Memory,
    ) as profiler_instance:
        idx, mask = gen_idx_sparse_mask(sparse_input)
        idx = idx.npu()

        print("sparse ffn")
        bench_profiler(run_sparse, total_iterations, profiler_instance)
        ys = run_sparse()

    verify_result(ys.cpu().float(), y.cpu().float(), 1e-2)


if __name__ == "__main__":
    for sparsity in SPARSITY_LEVELS:
        for model_name in MODEL_NAMES:
            config = ModelConfig(model_name)
            benchmark_ffn_profiler(sparsity, config)
