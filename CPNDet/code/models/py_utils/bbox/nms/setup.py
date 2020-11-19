import os
from setuptools import Extension, find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name='nms_wrapper',
    ext_modules=[
        CUDAExtension('nms_cpu', [
            'src/nms_cpu.cpp'
        ]),
        CUDAExtension('nms_cuda', [
            'src/nms_cuda.cpp',
            'src/nms_kernel.cu'
        ]),
    ],
    cmdclass={'build_ext': BuildExtension})
