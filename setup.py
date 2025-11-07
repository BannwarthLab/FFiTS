import os
from skbuild import setup
from setuptools import find_packages


lib_folder = os.path.dirname(os.path.realpath(__file__))
requirement_path = lib_folder + '/requirements.txt'


if os.path.isfile(requirement_path):
    with open(requirement_path) as f:
        install_requirements = list(f.read().splitlines())

setup(
    name='ffits',
    author='Daria Babushkina',
    author_email='babushkina@pc.rwth-aachen.de',
    version='0.8.0a',
    description='FFiTS: Force Field-interpolated Transition States',
    license='MIT',
    install_requires=install_requirements,
    # packages=["pytsguess", "pytsguess.force_field"],
    # cmake_install_dir="./src/force_field/",  
    cmake_args=['-DSKBUILD=ON'],
    classifiers=[
        "Development Status :: 1 - alpha",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License"
    ],
    entry_points={
        "console_scripts": [
            "ffits = ffits.main:main"
        ]
    },
)
