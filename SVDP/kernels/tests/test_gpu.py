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

import sparse_mlp_llama2
import sparse_mlp_mistral
import sparse_mlp_qwen2
import torch
import torch.profiler

torch.manual_seed(0)


# @torch.profiler.record_function("label")
@torch.inference_mode()
def dense_ffn(x, gate, up, down):
    g = gate @ x
    u = up @ x
    gu = torch.nn.functional.relu(g) * u
    return down @ gu


@torch.inference_mode()
def sparse_ffn(x, prediction, gate, up, down, sparse_module):
    # s1 = (prediction > 0).sum()
    sparse_module.sparse_gate_proj(gate, x, prediction)
    torch.cuda.synchronize()
    # s2 = (prediction > 0).sum()
    # assert s1==s2
    sparse_module.sparse_up_proj(up, x, prediction, 0.0)
    torch.cuda.synchronize()
    sparse_module.sparse_down_proj(down, prediction, x)
    torch.cuda.synchronize()
    return x


shapes = {
    "llama2": (sparse_mlp_llama2, [11008, 4096]),
    "mistral": (sparse_mlp_mistral, [14336, 4096]),
    "qwen2": (sparse_mlp_qwen2, [18944, 3584]),
}

DTYPE = torch.float16
DEVICE = "cuda"


def create_profiler(name, warmup_iterations, profile_iterations) -> torch.profiler.profile:
    return torch.profiler.profile(
        activities=[
            # torch.profiler.ProfilerActivity.CPU,
            torch.profiler.ProfilerActivity.CUDA,
        ],
        schedule=torch.profiler.schedule(
            wait=0,
            warmup=warmup_iterations,
            active=profile_iterations,
            repeat=1,
            skip_first=0,
        ),
        on_trace_ready=torch.profiler.tensorboard_trace_handler(f"perf_{name}"),
        profile_memory=True,
        # record_shapes=True,
        # with_stack=True
    )


def bench_profiler(fn, steps, profiler_instance):
    for _ in range(steps):
        fn()
        profiler_instance.step()
        torch.cuda.synchronize()


warmup_iterations, profile_iterations = 30 * 1, 20000
total_iterations = warmup_iterations + profile_iterations

for model_name, (sparse_module, (intermediate_size, hidden_size)) in shapes.items():
    for sparsity_ratio in (20, 50, 80, 95):
        x = 0.001 * torch.rand(hidden_size, 1).to(torch.float16).cuda()
        prediction = torch.ones(intermediate_size, dtype=DTYPE, device=DEVICE)
        prediction[torch.randperm(intermediate_size)[: int(intermediate_size * sparsity_ratio / 100)]] = -1
        gate = 0.001 * torch.rand(intermediate_size, hidden_size, dtype=DTYPE, device=DEVICE)
        up = 0.001 * torch.rand(intermediate_size, hidden_size, dtype=DTYPE, device=DEVICE)
        down = 0.001 * torch.rand(hidden_size, intermediate_size, dtype=DTYPE, device=DEVICE)
        gate[prediction <= 0] = 0.0
        up[prediction <= 0] = 0.0
        down[:, prediction <= 0] = 0.0
        down_t = down.t().contiguous()

        for _ in range(warmup_iterations):
            dense_ffn(x, gate, up, down)

        timings = []
        for _ in range(profile_iterations):
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            y = dense_ffn(x, gate, up, down)
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
        dense_time = torch.tensor(timings).mean().item()
        dense_time_std = torch.tensor(timings).std().item()
        print(f"{model_name} at {sparsity_ratio}% sparsity mean time (DENSE): {dense_time:.4f}+-{dense_time_std:.4f}ms")

        for _ in range(warmup_iterations):
            x_tmp = x.clone()
            prediction_tmp = prediction.clone()
            sparse_ffn(x_tmp, prediction_tmp, gate, up, down_t, sparse_module)

        timings = []
        for _ in range(profile_iterations):
            x_tmp = x.clone()
            prediction_tmp = prediction.clone()
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            sparse_ffn(x_tmp, prediction_tmp, gate, up, down_t, sparse_module)
            end_event.record()
            torch.cuda.synchronize()
            timings.append(start_event.elapsed_time(end_event))
        sparse_time = torch.tensor(timings).mean().item()
        sparse_time_std = torch.tensor(timings).std().item()
        print(
            f"{model_name} at {sparsity_ratio}% sparsity mean time (KERNEL): {sparse_time:.4f}+-{sparse_time_std:.4f}ms"
        )

        y_dense = dense_ffn(x, gate, up, down)
        x_tmp = x.clone()
        prediction_tmp = prediction.clone()
        sparse_ffn(x_tmp, prediction_tmp, gate, up, down_t, sparse_module)
        y_sparse = x_tmp
        assert torch.allclose(y_dense, y_sparse, atol=1e-6, rtol=1e-3), "Results differ significantly!"
        print(f"{model_name} {sparsity_ratio}% sparsity SPEEDUP: {dense_time / sparse_time:.4f}")
