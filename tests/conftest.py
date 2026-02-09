"""
Pytest configuration for loading required environment modules.

This module ensures that xtb and other required modules are properly
loaded before running tests, making tests work consistently in both
terminal and VSCode environments.
"""

import os
import subprocess
import sys
from pathlib import Path


def pytest_configure(config):
    """
    Load xtb module and set up environment before running tests.
    
    This hook is called after command line options have been parsed
    but before test collection begins.
    """
    # Try multiple methods to make xtb available
    xtb_found = False
    
    # Method 1: Check if xtb is already available
    if _is_xtb_available():
        xtb_found = True
        print("[conftest] xtb is available in PATH")
    
    # Method 2: Try to load xtb module
    if not xtb_found:
        if _load_xtb_module():
            xtb_found = True

    
    if xtb_found:
        print("[conftest] xtb is configured and ready")
    else:
        print("[conftest] Warning: xtb could not be found or configured", file=sys.stderr)


def _is_xtb_available():
    """Check if xtb is available in the system."""
    try:
        result = subprocess.run(
            ['which', 'xtb'],
            capture_output=True,
            text=True,
            timeout=5
        )
        return result.returncode == 0
    except (subprocess.TimeoutExpired, FileNotFoundError):
        return False


def _load_xtb_module():
    """
    Load the xtb module using system module loader.
    
    This sources the module environment and updates the current process's
    environment variables to make xtb available.
    """
    try:
        # Try to load xtb module via bash
        result = subprocess.run(
            ['bash', '-c', 'module load xtb && env'],
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            # Parse the environment from the output
            for line in result.stdout.split('\n'):
                if '=' in line:
                    key, value = line.split('=', 1)
                    os.environ[key] = value
            
            # Verify xtb is now available
            if _is_xtb_available():
                print("[conftest] Successfully loaded xtb module")
                return True
    except (subprocess.TimeoutExpired, FileNotFoundError, Exception) as e:
        pass  # Silent fail, try next method
    
    return False

