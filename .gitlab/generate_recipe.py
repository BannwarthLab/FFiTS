import sys
from typing import Dict, Tuple

# Configuration mappings for each component
OS_CONFIG = {
    "rl9": "rockylinux:9",
    "rl10": "rockylinux/rockylinux:10",
}

COMPILER_CONFIG = {
    "gnu": {
        "packages": "gcc-gfortran",
        "env": {"CC": "gcc", "CXX": "g++", "FC": "gfortran", "F77": "gfortran", "F90": "gfortran"},
    },
    "intel2023.2.0": {
        "repo": True,
        "packages": "intel-oneapi-compiler-dpcpp-cpp-2023.2.0 intel-oneapi-compiler-fortran-2023.2.0",
        "env": {
            "CC": "icx", "CXX": "icpx", "FC": "ifx", "F77": "ifx", "F90": "ifx",
            "PATH": "/opt/intel/oneapi/compiler/2023.2.0/linux/bin:$PATH",
            "TBBROOT": "/opt/intel/oneapi/tbb/2021.10.0",
            "COMPILERROOT": "/opt/intel/oneapi/compiler/2023.2.0",
        },
    },
    "intel2025.3.0": {
        "repo": True,
        "packages": "intel-oneapi-compiler-dpcpp-cpp-2025.3.0 intel-oneapi-compiler-fortran-2025.3.0",
        "env": {
            "CC": "icx", "CXX": "icpx", "FC": "ifx", "F77": "ifx", "F90": "ifx",
            "PATH": "/opt/intel/oneapi/compiler/2025.3.0/linux/bin:$PATH",
            "TBBROOT": "/opt/intel/oneapi/tbb/2022.3",
            "COMPILERROOT": "/opt/intel/oneapi/compiler/2025.3",
        },
    },
}

# Python versions available directly via dnf/EPEL on RockyLinux 9/10.
# NOTE: 3.9 is RL9's system python and is always present; RL10 ships a newer
# default but the versioned packages below should still be installable via
# AppStream/EPEL. If a given OS+version combo isn't packaged, that cell of
# the matrix will need to be dropped or use the deadsnakes-equivalent COPR.
PYTHON_CONFIG = {
    "python3.9": "python3.9 python3.9-devel python3.9-pip",
    "python3.10": "python3.10 python3.10-devel python3.10-pip",
    "python3.11": "python3.11 python3.11-devel python3.11-pip",
    "python3.12": "python3.12 python3.12-devel python3.12-pip",
    "python3.13": "python3.13 python3.13-devel python3.13-pip",
    "python3.14": "python3.14 python3.14-devel python3.14-pip",
}

XTB_VERSION = "6.7.1"


def parse_recipe_name(name: str) -> Tuple[str, str, str]:
    """Parse recipe name in format: OS-COMPILER-PYTHON"""
    parts = name.lower().split("-")
    if len(parts) != 3:
        raise ValueError(f"Invalid recipe name format. Expected: OS-COMPILER-PYTHON, got: {name}")
    return tuple(parts)


def validate_config(os_name: str, compiler: str, python: str) -> None:
    """Validate that all configuration values are supported."""
    if os_name not in OS_CONFIG:
        raise ValueError(f"Unknown OS: {os_name} (supported: {', '.join(OS_CONFIG.keys())})")
    if compiler not in COMPILER_CONFIG:
        raise ValueError(f"Unknown compiler: {compiler} (supported: {', '.join(COMPILER_CONFIG.keys())})")
    if python not in PYTHON_CONFIG:
        raise ValueError(f"Unknown python: {python} (supported: {', '.join(PYTHON_CONFIG.keys())})")


def generate_recipe(name: str) -> str:
    """Generate the Apptainer recipe file content."""
    os_name, compiler, python = parse_recipe_name(name)
    validate_config(os_name, compiler, python)

    lines = []

    # Bootstrap section
    lines.append("Bootstrap: docker")
    lines.append(f"From: {OS_CONFIG[os_name]}\n")

    # Post section
    lines.append("%post")
    lines.append("    # Update system and install necessary packages")
    lines.append("    export DEBIAN_FRONTEND=noninteractive")
    lines.append("    export TZ=Europe/Berlin")
    lines.append("    dnf -y update")
    lines.append("    dnf groupinstall -y \"Development Tools\"")
    lines.append("    dnf install -y wget git cmake")
    lines.append("    dnf install -y epel-release")
    lines.append("    dnf config-manager --set-enabled crb")
    lines.append("")

    # Intel OneAPI repo if needed
    if COMPILER_CONFIG[compiler].get("repo"):
        lines.append("    # Intel OneAPI Repository")
        lines.append("    dnf install -y procps-ng")
        lines.append("    tee > /tmp/oneAPI.repo << 'EOF'")
        lines.append("[oneAPI]")
        lines.append("name=Intel® oneAPI repository")
        lines.append("baseurl=https://yum.repos.intel.com/oneapi")
        lines.append("enabled=1")
        lines.append("gpgcheck=1")
        lines.append("repo_gpgcheck=1")
        lines.append("gpgkey=https://yum.repos.intel.com/intel-gpg-keys/GPG-PUB-KEY-INTEL-SW-PRODUCTS.PUB")
        lines.append("EOF")
        lines.append("    mv /tmp/oneAPI.repo /etc/yum.repos.d")
        lines.append("")

    # Compiler installation
    lines.append(f"    # {compiler.title()} Compiler")
    lines.append(f"    dnf install -y {COMPILER_CONFIG[compiler]['packages']}")
    lines.append("")

    # Python installation
    lines.append(f"    # {python.title()} Installation")
    lines.append(f"    dnf install -y {PYTHON_CONFIG[python]}")
    lines.append(f"    ln -sf /usr/bin/{python} /usr/local/bin/python")
    lines.append(f"    ln -sf /usr/bin/{python} /usr/local/bin/python3")
    lines.append(f"    ln -sf /usr/bin/{python.replace('python', 'pip')} /usr/local/bin/pip")
    lines.append(f"    ln -sf /usr/bin/{python.replace('python', 'pip')} /usr/local/bin/pip3")
    lines.append("")

    # xtb installation (pinned version)
    lines.append(f"    # xtb {XTB_VERSION} Installation")
    lines.append(f"    wget https://github.com/grimme-lab/xtb/releases/download/v{XTB_VERSION}/xtb-{XTB_VERSION}-linux-x86_64.tar.xz")
    lines.append(f"    tar -xvf xtb-{XTB_VERSION}-linux-x86_64.tar.xz")
    lines.append("    chmod 777 -R xtb-dist")
    lines.append("")

    # Cleanup
    lines.append("    # Clean up cache to reduce image size")
    lines.append("    dnf clean all")
    lines.append("    rm -rf /var/cache/dnf\n")

    # Environment section
    lines.append("%environment")
    lines.append("    export PATH=\"/root/.local/bin:/usr/local/bin:/usr/bin:${PATH}\"")
    lines.append("    export XTBEXE=/xtb-dist/bin/")
    lines.append("    export PATH=$XTBEXE:$PATH")

    env_vars = {}
    env_vars.update(COMPILER_CONFIG[compiler].get("env", {}))
    for key, value in env_vars.items():
        lines.append(f"    export {key}={value}")

    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_recipe.py <recipe-name>")
        print("Format: OS-COMPILER-PYTHON")
        print(f"Supported OS: {', '.join(OS_CONFIG.keys())}")
        print(f"Supported Compilers: {', '.join(COMPILER_CONFIG.keys())}")
        print(f"Supported Python: {', '.join(PYTHON_CONFIG.keys())}")
        sys.exit(1)

    name = sys.argv[1]
    print(f"Making recipe for {name}")

    try:
        content = generate_recipe(name)
        with open(f"{name}.def", "w") as f:
            f.write(content)
        print(f"Successfully created {name}.def")
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)