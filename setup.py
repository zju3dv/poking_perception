#!/usr/bin/env python
from setuptools import find_packages
from setuptools import setup

setup(
    name="puop",
    version="0.1",
    author="chenlinghao",
    packages=find_packages(exclude=("configs", "tests",)),
)
