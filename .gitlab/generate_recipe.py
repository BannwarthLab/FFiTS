import sys
from typing import Dict, Tuple

# Configuration mappings for each component
OS_CONFIG = {
    "rl9": "rockylinux:9",
    "rl10": "rockylinux/rockylinux:10",
}

PYTHON_CONFIG = {
    "none": {},
    "python3.12": "python3.12 python3.12-devel python3.12-pip",
    "python3.13": "python3.13 python3.13-devel python3.13-pip",
    "python3.14": "python3.14 python3.14-devel python3.14-pip",
}

def parse_recipe_name(name: str) -> Tuple[str, str]:
    """Parse recipe name in format: OS-PYTHON"""
    parts = name.lower().split("-")
    if len(parts) != 2:
        raise ValueError(f"Invalid recipe name format. Expected: OS-PYTHON, got: {name}")
    return tuple(parts)


def validate_config(os_name: str, python: str) -> None:
    """Validate that all configuration values are supported."""
    if os_name not in OS_CONFIG:
        raise ValueError(f"Unknown OS: {os_name} (supported: {', '.join(OS_CONFIG.keys())})")
    if python not in PYTHON_CONFIG:
        raise ValueError(f"Unknown python: {python} (supported: {', '.join(PYTHON_CONFIG.keys())})")


def generate_recipe(name: str) -> str:
    """Generate the Apptainer recipe file content."""
    os_name, python = parse_recipe_name(name)
    validate_config(os_name, python)

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
    lines.append("    dnf install -y wget git")
    lines.append("    dnf install -y epel-release")
    lines.append("    dnf config-manager --set-enabled crb")
    lines.append("")
    
    # Python installation
    if python != "none":
        lines.append(f"    # {python.title()} Installation")
        lines.append(f"    dnf install -y {PYTHON_CONFIG[python]}")
        lines.append(f"    ln -sf /usr/bin/{python} /usr/local/bin/python")
        lines.append(f"    ln -sf /usr/bin/{python} /usr/local/bin/python3")
        lines.append(f"    ln -sf /usr/bin/{python.replace('python', 'pip')} /usr/local/bin/pip")
        lines.append(f"    ln -sf /usr/bin/{python.replace('python', 'pip')} /usr/local/bin/pip3")
        lines.append("")

    # Cleanup
    lines.append("    # Clean up cache to reduce image size")
    lines.append("    dnf clean all")
    lines.append("    rm -rf /var/cache/dnf\n")
    
    # Environment section
    lines.append("%environment")
    lines.append("    export PATH=\"/root/.local/bin:/usr/local/bin:/usr/bin:${PATH}\"")
    
    return "\n".join(lines)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python generate_recipe.py <recipe-name>")
        print("Format: OS-PYTHON")
        print(f"Supported OS: {', '.join(OS_CONFIG.keys())}")
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