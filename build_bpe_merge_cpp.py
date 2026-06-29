from setuptools import setup, Extension
import pybind11

ext_modules = [
    Extension(
        "bpe_merge",
        ["bpe_merge.cpp"],
        include_dirs=[
            pybind11.get_include(),
        ],
        language="c++",
        extra_compile_args=[
            "-O3",
            "-std=c++17",
        ],
    ),
]

setup(
    name="bpe_merge",
    ext_modules=ext_modules,
)
