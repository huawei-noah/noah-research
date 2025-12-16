# -*- coding: utf-8 -*-
# Copyright (c) 2025 Huawei Technologies Co., Ltd. All Rights Reserved.

from dataclasses import dataclass

SYSTEM_PROMPT = """
You write custom Triton kernels to replace the pytorch operators in the given architecture to get speedups.
You have complete freedom to choose the set of operators you want to replace. 
You may make the decision to replace some operators with custom Triton kernels and leave others unchanged. 
You may replace multiple operators with custom implementations, consider operator fusion opportunities (combining multiple operators into a single kernel, for example, combining matmul+relu), or algorithmic changes (such as online softmax). 
You are only limited by your imagination.
Here's an example to show you the syntax of inline embedding custom operators from the Triton DSL in torch: The example given architecture is:
```
import torch
import torch.nn as nn
import torch.nn.functional as F


class Model(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        return a + b


def get_inputs():
    # randomly generate input tensors based on the model architecture
    a = torch.randn(1, 128).cuda()
    b = torch.randn(1, 128).cuda()
    return [a, b]


def get_init_inputs():
    # randomly generate tensors required for initialization based on the model architecture
    return []

```
The example new arch with custom Triton kernels looks like this:
```
import torch
import torch.nn as nn
import torch.nn.functional as F
import triton
import triton.language as tl


@triton.jit
def add_kernel(
    x_ptr,  # Pointer to first input
    y_ptr,  # Pointer to second input
    out_ptr,  # Pointer to output
    n_elements,  # Total number of elements in input/output
    BLOCK_SIZE: tl.constexpr,
):
    # Each program handles a contiguous block of data of size BLOCK_SIZE
    block_start = tl.program_id(0) * BLOCK_SIZE
    # Create a range of offsets [0..BLOCK_SIZE-1]
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Mask to ensure we don't go out of bounds
    mask = offsets < n_elements
    # Load input values
    x = tl.load(x_ptr + offsets, mask=mask, other=0.0)
    y = tl.load(y_ptr + offsets, mask=mask, other=0.0)
    # Perform the elementwise addition
    out = x + y
    # Store the result
    tl.store(out_ptr + offsets, out, mask=mask)


def triton_add(x: torch.Tensor, y: torch.Tensor):
    \"\"\"
    This function wraps the Triton kernel call. It:
      1. Ensures the inputs are contiguous on GPU.
      2. Calculates the grid (blocks) needed.
      3. Launches the Triton kernel.
    \"\"\"
    assert x.is_cuda and y.is_cuda, "Tensors must be on CUDA."
    x = x.contiguous()
    y = y.contiguous()

    # Prepare output tensor
    out = torch.empty_like(x)

    # Number of elements in the tensor
    n_elements = x.numel()
    BLOCK_SIZE = 128  # Tunable parameter for block size

    # Determine the number of blocks needed
    grid = lambda meta: ((n_elements + meta["BLOCK_SIZE"] - 1) // meta["BLOCK_SIZE"],)

    # Launch the Triton kernel
    add_kernel[grid](x, y, out, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    return out


class ModelNew(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, a, b):
        # Instead of "return a + b", call our Triton-based addition
        return triton_add(a, b)

```
"""

USER_PROMPT = """
You are given the following architecture:
```
{original_code}
```

# Validation and output protocol (strictly required)
1. You should first generate the candidate ModelNew source code as a string (complete, compilable code), but **do not output it yet**.
2. Call the tool: `metrics = validate_model(candidate_code)`.
3. Based on the returned `metrics`, make decisions:
   - If `metrics.error` is not None → fix the issue (e.g., compilation or runtime error), regenerate the code, and re-run validation. Retry up to **3 times**.
   - If `metrics.error` is None but `metrics.speedup <= 1.0` → attempt further optimizations (e.g., operator fusion, kernel parameter tuning, algorithmic improvements) and re-run validation. Retry up to **2 times**.
   - If all retries fail to achieve both correctness and speedup > 1.0, output a single-line failure message and stop.
4. Only when `metrics.error == None` **and** `metrics.speedup > 1.0` should you output the final `ModelNew` code in a **single code block**.
5. When outputting the code, you must print **only** the code block — no explanations, metrics, or validation logs.
"""


@dataclass
class PromptSampler:
    system_prompt: str
    user_prompt: str

    def __init__(self, initial_code: str):
        self.system_prompt = SYSTEM_PROMPT
        self.user_prompt = USER_PROMPT.format(original_code=initial_code)