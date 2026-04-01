import os
import sys

# Detect if this is a build_ext --inplace call (developer workflow)
is_build_ext_inplace = "build_ext" in sys.argv and "--inplace" in sys.argv

if is_build_ext_inplace:
    # For local development, use scikit-build to compile the Fortran extension
    try:
        from skbuild import setup

        # Use skbuild for CMake-based Fortran build
        setup(
            cmake_args=["-DSKBUILD=ON"],
        )
    except ImportError:
        raise RuntimeError(
            "scikit-build is required for build_ext --inplace. Install with: pip install scikit-build"
        )
else:
    # For pip install, use plain setuptools (assumes .so already compiled)
    # All configuration is read from pyproject.toml
    from setuptools import setup

    setup()
