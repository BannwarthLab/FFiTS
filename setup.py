import os
from skbuild import setup
from setuptools import find_packages


lib_folder = os.path.dirname(os.path.realpath(__file__))


setup(
    name='pyTSguess',
    author='Daria Babushkina',
    author_email='babushkina@pc.rwth-aachen.de',
    # packages=["pytsguess", "pytsguess.force_field"],
    # cmake_install_dir="./src/force_field/",  
    cmake_args=['-DSKBUILD=ON'],
)