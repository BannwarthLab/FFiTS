import os

# For Python 3.11+, tomllib is built in.
# For older versions, install tomli (`pip install tomli`)
try:
    import tomllib
except ModuleNotFoundError:
    import tomli as tomllib


def load_toml(path: str) -> dict:
    """Load a TOML file from the given path."""
    with open(path, "rb") as f:
        return tomllib.load(f)


def deep_update(base: dict, updates: dict) -> dict:
    """
    Recursively update a nested dict (base) with another (updates).
    Keeps unspecified defaults while overwriting provided keys.
    """
    for key, value in updates.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            deep_update(base[key], value)
        else:
            base[key] = value
    return base


def load_config(user_path: str | None = None) -> dict:
    """Load the default TOML config and update with user overrides."""
    current_dir = os.getcwd()
    default_path = os.path.join(current_dir, 'src', 'data', 'default_config.toml')

    # Load defaults
    config = load_toml(default_path)

    # Load and merge user config if provided
    if user_path and os.path.exists(user_path):
        user_config = load_toml(user_path)
        config = deep_update(config, user_config)
    else:
        print("[INFO] No user config found, using defaults.")

    return config
