"""
Pytest configuration for loading required environment modules.

This module ensures that xtb and other required modules are properly
loaded before running tests, making tests work consistently in both
terminal and VSCode environments.
"""

import os
import subprocess
import sys
import ctypes
import shutil


def pytest_configure(config):
    """
    Load xtb module and set up environment before running tests.

    This hook is called after command line options have been parsed
    but before test collection begins.
    """
    if _load_module("compiler"):
        _preload_known_compiler_runtimes()
    else:
        print("[conftest] Warning: could not load compiler module", file=sys.stderr)

    if not _is_xtb_available():
        _load_module("xtb")

    if _is_xtb_available():
        print("[conftest] xtb is configured and ready")
    else:
        print(
            "[conftest] Warning: xtb could not be found or configured", file=sys.stderr
        )


def _is_xtb_available():
    """Check if xtb is available in the system."""
    return shutil.which("xtb") is not None


def _load_module(module_name):
    """
    Load an environment module using the system module loader.

    This uses a login bash shell and common module init scripts so it works
    in non-interactive pytest runs (e.g. VS Code test discovery on clusters).
    """
    cmd = f"module load {module_name} && env"

    try:
        result = subprocess.run(
            ["bash", "-ilc", cmd],
            capture_output=True,
            text=True,
            timeout=10,
        )

        if result.returncode == 0:
            for line in result.stdout.split("\n"):
                if "=" in line:
                    key, value = line.split("=", 1)
                    os.environ[key] = value

            if module_name == "xtb" and _is_xtb_available():
                print("[conftest] Successfully loaded xtb module")
                return True
            if module_name != "xtb":
                print(f"[conftest] Successfully loaded {module_name} module")
                return True

        error_detail = (result.stderr or result.stdout).strip()
        if error_detail:
            print(
                f"[conftest] module load {module_name} failed: {error_detail}",
                file=sys.stderr,
            )
    except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
        print(
            f"[conftest] module load {module_name} failed: {exc}",
            file=sys.stderr,
        )
    except Exception as exc:
        print(
            f"[conftest] unexpected error while loading {module_name}: {exc}",
            file=sys.stderr,
        )

    return False


def _preload_known_compiler_runtimes():
    """Preload common Intel runtime libs needed by the Fortran extension."""
    runtime_libs = [
        "libintlc.so.5",
        "libimf.so",
        "libsvml.so",
        "libifcoremt.so.5",
        "libifport.so.5",
        "libirng.so",
    ]

    search_dirs = []
    for root_env in ("CMPLR_ROOT", "ONEAPI_ROOT"):
        root = os.environ.get(root_env, "")
        if root:
            for suffix in ("lib", "lib/intel64", "linux/compiler/lib/intel64_lin"):
                candidate = os.path.join(root, suffix)
                if os.path.isdir(candidate) and candidate not in search_dirs:
                    search_dirs.append(candidate)

    for env_name in ("LD_LIBRARY_PATH", "LIBRARY_PATH"):
        for part in os.environ.get(env_name, "").split(":"):
            if part and part not in search_dirs:
                search_dirs.append(part)

    loaded_any = False
    for lib_name in runtime_libs:
        lib_path = None
        for directory in search_dirs:
            candidate = os.path.join(directory, lib_name)
            if os.path.isfile(candidate):
                lib_path = candidate
                break

        if not lib_path:
            continue

        try:
            ctypes.CDLL(lib_path, mode=ctypes.RTLD_GLOBAL)
            loaded_any = True
        except OSError as exc:
            print(
                f"[conftest] failed to preload {lib_name}: {exc}",
                file=sys.stderr,
            )

    if loaded_any:
        print("[conftest] Preloaded compiler runtime libraries")
