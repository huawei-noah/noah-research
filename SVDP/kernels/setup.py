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

import os

from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# GUIDE
# `python setup.py install` or `pip install .`
# `import my_extension`

# Identify GPU compute architectures
compute_archs = os.environ.get("CUDA_ARCH_LIST", None)
nvcc_flags = ["-O3", "--use_fast_math"]
if compute_archs is not None:
    nvcc_flags.extend([f"-gencode=arch=compute_{arch},code=sm_{arch}" for arch in compute_archs.split(",")])

llm_arch = os.environ.get("LLM_ARCH", "llama2")
assert llm_arch in ("llama2", "qwen2", "mistral")

nvcc_flags.append(f"-D{llm_arch.upper()}")

setup(
    name=f"sparse_mlp_{llm_arch}",
    version="1.0.0",
    description="A module with 3 CUDA kernels for sparse MLP inference",
    ext_modules=[
        CUDAExtension(
            name=f"sparse_mlp_{llm_arch}",
            verbose=True,
            sources=[
                "src/sparse_gate_proj.cpp",
                "src/sparse_gate_proj_kernel.cu",
                "src/sparse_up_proj.cpp",
                "src/sparse_up_proj_kernel.cu",
                "src/sparse_up_proj_drelu.cpp",
                "src/sparse_up_proj_drelu_kernel.cu",
                "src/sparse_down_proj.cpp",
                "src/sparse_down_proj_kernel.cu",
                "src/bindings.cpp",
            ],
            extra_compile_args={"cxx": ["-O3"], "nvcc": nvcc_flags},
        )
    ],
    cmdclass={"build_ext": BuildExtension},
)
