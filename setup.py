"""
Builds the `gesture_native` C++ extension.

Usage:
    cd native
    pip install .

This compiles gesture_engine.cpp (pybind11) into an importable Python
extension module and installs it into the active environment.
"""
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

ext_modules = [
    Pybind11Extension(
        "gesture_native",
        ["gesture_engine.cpp"],
        cxx_std=17,
    ),
]

setup(
    name="gesture_native",
    version="1.0.0",
    description="Native hot-loop helpers for the AI Gesture Command Center",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False,
)
