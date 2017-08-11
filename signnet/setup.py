# -*- coding: utf-8 -*-

from setuptools import find_packages
from setuptools import setup

REQUIRED_PACKAGES = [
        'numpy>=1.13.1',
        'sklearn',
        'matplotlib>=2.0.2',
]

setup(
    name='signnet',
    version='0.1',
    install_requires=REQUIRED_PACKAGES,
    packages=find_packages(),
    include_package_data=True,
    requires=[]
)
