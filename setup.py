"""
Build script for Adaptive HNSW C++ extension.

Prerequisites:
    pip install pybind11
    Visual Studio Build Tools (Windows) or g++ (Linux/Mac)

Build:
    python setup.py build_ext --inplace
"""

import os
from setuptools import setup
from pybind11.setup_helpers import Pybind11Extension, build_ext

extra_args = ["/O2", "/std:c++17"] if os.name == "nt" else ["-O3", "-std=c++17"]

setup(
    name="adaptive_hnsw_cpp",
    ext_modules=[
        Pybind11Extension(
            "adaptive_hnsw_cpp",
            ["bindings.cpp"],
            include_dirs=["."],
            extra_compile_args=extra_args,
        ),
    ],
    cmdclass={"build_ext": build_ext},
)
