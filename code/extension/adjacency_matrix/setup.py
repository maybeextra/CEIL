
from setuptools import setup

from torch.utils.cpp_extension import BuildExtension, CUDAExtension


setup(
    name='build_adjacency_matrix',
    ext_modules=[
        CUDAExtension('build_adjacency_matrix', [
            'build_adjacency_matrix.cpp',
            'build_adjacency_matrix_kernel.cu',
        ]),
    ],
    cmdclass={
        'build_ext':BuildExtension  
    })
