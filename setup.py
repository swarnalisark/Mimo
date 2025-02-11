#!/usr/bin/env python

from setuptools import find_packages, setup

with open("requirements.txt") as reqs_file:
    requirements = [req for req in reqs_file.read().splitlines() if not req.startswith(("#", "-"))]

setup(
    name="src",
    version="0.0.1",
    description="Repo for vision assisted beamforming",
    author="Jiekun_Jiale",
    author_email="jiekun96@gmail.com",
    python_requires=">=3.7",
    install_requires=requirements,
    packages=find_packages(),
)