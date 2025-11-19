import os
import sys
import subprocess
from setuptools import setup, find_packages

# Detect if this is a build_ext --inplace call (developer workflow)
is_build_ext_inplace = 'build_ext' in sys.argv and '--inplace' in sys.argv

if is_build_ext_inplace:
    # For local development, use scikit-build to compile the Fortran extension
    try:
        from skbuild import setup as skbuild_setup
        lib_folder = os.path.dirname(os.path.realpath(__file__))
        requirement_path = lib_folder + '/requirements.txt'
        
        if os.path.isfile(requirement_path):
            with open(requirement_path) as f:
                install_requirements = list(f.read().splitlines())
        else:
            install_requirements = []
        
        # Use skbuild for CMake-based Fortran build
        skbuild_setup(
            name='ffits',
            author='Daria Babushkina',
            author_email='babushkina@pc.rwth-aachen.de',
            version='0.8.0a',
            description='FFiTS: Force Field-interpolated Transition States',
            license='MIT',
            install_requires=install_requirements,
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
    except ImportError:
        raise RuntimeError("scikit-build is required for build_ext --inplace. Install with: pip install scikit-build")
else:
    # For pip install, use plain setuptools (assumes .so already compiled)
    lib_folder = os.path.dirname(os.path.realpath(__file__))
    requirement_path = lib_folder + '/requirements.txt'
    
    if os.path.isfile(requirement_path):
        with open(requirement_path) as f:
            install_requirements = list(f.read().splitlines())
    else:
        install_requirements = []
    
    setup(
        name='ffits',
        author='Daria Babushkina',
        author_email='babushkina@pc.rwth-aachen.de',
        version='0.8.0a',
        description='FFiTS: Force Field-interpolated Transition States',
        license='MIT',
        install_requires=install_requirements,
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

