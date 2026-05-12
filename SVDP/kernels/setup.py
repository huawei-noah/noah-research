from setuptools import setup
import os

from torch.utils.cpp_extension import BuildExtension, CUDAExtension

# GUIDE
# `python setup.py install` or `pip install .`
# `import my_extension`

# Identify GPU compute architectures
compute_archs = os.environ.get("CUDA_ARCH_LIST", None)
nvcc_flags = ['-O3', '--use_fast_math']
if compute_archs is not None:
    nvcc_flags.extend([f'-gencode=arch=compute_{arch},code=sm_{arch}' for arch in compute_archs.split(",")])

llm_arch = os.environ.get("LLM_ARCH", "llama2")
assert llm_arch in ("llama2", "qwen2", "mistral")

nvcc_flags.append(f'-D{llm_arch.upper()}')

setup(
    name=f'sparse_mlp_{llm_arch}',
    version='1.0.0',
    description='A module with 3 CUDA kernels for sparse MLP inference',
    ext_modules=[
        CUDAExtension(
            name=f'sparse_mlp_{llm_arch}',
            verbose=True,
            sources=[
                'src/sparse_gate_proj.cpp', 'src/sparse_gate_proj_kernel.cu',
                'src/sparse_up_proj.cpp', 'src/sparse_up_proj_kernel.cu',
                'src/sparse_up_proj_drelu.cpp', 'src/sparse_up_proj_drelu_kernel.cu',
                'src/sparse_down_proj.cpp', 'src/sparse_down_proj_kernel.cu',
                'src/bindings.cpp'
            ],
            extra_compile_args={
                'cxx': ['-O3'],
                'nvcc':  nvcc_flags 
            }
        )
    ],
    cmdclass={'build_ext': BuildExtension}
)