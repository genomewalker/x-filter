# x_filter/config.py

import os
from typing import Dict, Any, Optional
import yaml
import json

# Default configuration
DEFAULT_CONFIG = {
    "threads": 1,
    "max_memory": "4G",
    "evalue": 1e-10,
    "bitscore": 60,
    "chunk_size": 10_000_000,
    "filters": {
        "breadth": 0.5,
        "depth": 0.1,
        "depth_evenness": 1.0,
        "breadth_expected_ratio": 0.5,
    },
    "reassignment": {
        "iters": 25,
        "scale": 0.9,
        "step_min": -1.0,
        "step_max": -1.0,
        "mstep": 4,
    },
    "trim": {"enabled": True, "multiplier": 2.0, "offset": 10},
}


def parse_memory_limit(memory_limit: str) -> int:
    """Convert memory limit string to bytes."""
    units = {"B": 1, "K": 1024, "M": 1024**2, "G": 1024**3, "T": 1024**4}

    memory_limit = memory_limit.upper().strip()
    if memory_limit[-1] in units:
        value = float(memory_limit[:-1])
        unit = memory_limit[-1]
        return int(value * units[unit])
    else:
        return int(memory_limit)


# x_filter/config.py (continued)
def format_memory_size(size_in_bytes: int) -> str:
    """Format memory size in appropriate units."""
    if size_in_bytes >= 1024**3:
        return f"{size_in_bytes / 1024**3:.2f}G"
    elif size_in_bytes >= 1024**2:
        return f"{size_in_bytes / 1024**2:.2f}M"
    elif size_in_bytes >= 1024:
        return f"{size_in_bytes / 1024:.2f}K"
    else:
        return f"{size_in_bytes}B"


class Config:
    """Configuration management for xFilter."""

    def __init__(self, config_file: Optional[str] = None, **overrides):
        """
        Initialize configuration.

        Args:
            config_file: Optional path to configuration file
            overrides: Keyword arguments to override configuration
        """
        # Start with default configuration
        self.config = DEFAULT_CONFIG.copy()

        # Load from file if provided
        if config_file and os.path.exists(config_file):
            self._load_from_file(config_file)

        # Apply overrides
        self._apply_overrides(overrides)

        # Validate configuration
        self._validate()

    def _load_from_file(self, config_file: str) -> None:
        """Load configuration from file."""
        _, ext = os.path.splitext(config_file)

        with open(config_file, "r") as f:
            if ext.lower() == ".yaml" or ext.lower() == ".yml":
                loaded_config = yaml.safe_load(f)
            elif ext.lower() == ".json":
                loaded_config = json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")

        # Update configuration
        self._update_config(loaded_config)

    def _apply_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply configuration overrides."""
        self._update_config(overrides)

    def _update_config(self, updates: Dict[str, Any]) -> None:
        """Update configuration with nested dictionaries."""

        def update_nested(base, updates):
            for key, value in updates.items():
                if (
                    key in base
                    and isinstance(base[key], dict)
                    and isinstance(value, dict)
                ):
                    update_nested(base[key], value)
                else:
                    base[key] = value

        update_nested(self.config, updates)

    def _validate(self) -> None:
        """Validate configuration values."""
        # Convert memory string to bytes if needed
        if isinstance(self.config["max_memory"], str):
            self.config["max_memory"] = parse_memory_limit(self.config["max_memory"])

        # Ensure threads is an integer
        self.config["threads"] = int(self.config["threads"])

        # Validate filters
        for filter_name, threshold in self.config["filters"].items():
            if filter_name not in [
                "breadth",
                "depth",
                "depth_evenness",
                "breadth_expected_ratio",
            ]:
                raise ValueError(f"Invalid filter name: {filter_name}")

            if not 0 <= threshold <= 1 and filter_name != "depth":
                raise ValueError(
                    f"Filter threshold must be between 0 and 1: {filter_name}={threshold}"
                )

    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        if "." in key:
            # Handle nested keys
            parts = key.split(".")
            value = self.config
            for part in parts:
                if part not in value:
                    return default
                value = value[part]
            return value

        return self.config.get(key, default)

    def __getitem__(self, key: str) -> Any:
        """Get configuration value using dictionary syntax."""
        return self.get(key)

    def update(self, updates: Dict[str, Any]) -> None:
        """Update configuration."""
        self._update_config(updates)
        self._validate()

    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return self.config.copy()

    def save(self, filename: str) -> None:
        """Save configuration to file."""
        _, ext = os.path.splitext(filename)

        with open(filename, "w") as f:
            if ext.lower() == ".yaml" or ext.lower() == ".yml":
                yaml.dump(self.config, f, default_flow_style=False)
            elif ext.lower() == ".json":
                json.dump(self.config, f, indent=2)
            else:
                raise ValueError(f"Unsupported configuration file format: {ext}")
