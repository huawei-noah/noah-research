import os
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='roi_align',
    ext_modules=[
        CUDAExtension('roi_align_cuda', [
            'src/roi_align_cuda.cpp',
            'src/roi_align_kernel.cu',
        ])
    ],
    cmdclass={'build_ext': BuildExtension})
