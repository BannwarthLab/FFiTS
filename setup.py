import os
from skbuild import setup
from setuptools import find_packages



setup(
    name='pyTSguess',
    author='Daria Babushkina',
    author_email='babushkina@pc.rwth-aachen.de',
    cmake_args=['-DSKBUILD=ON'],
)