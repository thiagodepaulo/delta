import argparse
import json
import yaml
from typing import Any
from delta.configs.dataset import DatasetConfig

def _load_config_from_single_file(filepath: str) -> dict:
    """Load a config dictionary from a file. Supports YAML and JSON."""
    with open(filepath, "r") as f:
        if filepath.endswith(".json"):
            return json.load(f)
        elif filepath.endswith(".yaml") or filepath.endswith(".yml"):
            return yaml.safe_load(f)
        else:
            raise ValueError(f"Unsupported file extension: {filepath}")

        
def load_config(*paths: str, cls: type | None = None) -> Any:
    """Load a config from a list of filepaths.

    Configs from multiple files are merged sequentially,
    with later files overriding earlier ones for duplicate keys.

    If cls is provided, the config is returned as an instance of cls.
    Otherwise, the config is returned as a dictionary.
    """

    if not paths:
        raise ValueError("At least one path is required")
    config = {}
    for filepath in paths:
        new_config = _load_config_from_single_file(filepath)
        config = deep_merge_dict(config, new_config)

    return cls(**config) if cls else config        

def load_config_from_namespace(
    args: argparse.Namespace, cls: type | None = None
) -> Any:
    config_dict = {}

    if args.config_paths:
        config_dict = load_config(*args.config_paths)

    filepath_args = set()
    user_provided_args = {}

    for arg, value in vars(args).items():
        if arg in ["config_paths", "at_least_one"] or value is None:
            continue
        if arg.startswith(".config_path_"):
            arg = arg.removeprefix(".config_path_")
            filepath_args.add(arg)
        user_provided_args[arg] = value

    for arg, value in sorted(user_provided_args.items()):
        *prefix_keys, current_key = arg.split(".")
        if arg in filepath_args:
            value = load_config(value)

        current = config_dict
        for key in prefix_keys:
            if key not in current or current[key] is None:
                current[key] = {}
            current = current[key]
        current[current_key] = value

    # Let Pydantic handle the type conversion
    return cls(**config_dict) if cls else config_dict

class ListUpdateStrategy:
    """Strategy for overriding lists in deep_merge_list."""

    APPEND = "append"
    PREPEND = "prepend"
    REPLACE = "replace"
    FULL_REPLACE = "full_replace"
    RECURSIVE_REPLACE = "recursive_replace"

def deep_merge(
    base: Any,
    override: Any,
    list_update_strategy: str = ListUpdateStrategy.FULL_REPLACE,
) -> Any:
    if isinstance(base, dict) and isinstance(override, dict):
        return deep_merge_dict(base, override, list_update_strategy)
    elif isinstance(base, list) and isinstance(override, list):
        return deep_merge_list(base, override, list_update_strategy)
    return override


def deep_merge_dict(
    base: dict,
    override: dict,
    list_update_strategy: str = ListUpdateStrategy.FULL_REPLACE,
) -> dict:
    """
    Deep merges two dictionaries recursively.

    Args:
        base (dict): The base dictionary.
        override (dict): The dictionary to override the base dictionary.

    Returns:
        dict: The merged dictionary.

    """
    result = base.copy()
    for key, value in override.items():
        if key in result:
            result[key] = deep_merge(result[key], value, list_update_strategy)
        else:
            result[key] = value
    return result

#def load_data_config(dts_config_file):
#    with open(dts_config_file, "r") as f:
#        dc_ = yaml.safe_load(f)
#        dataset_cfg = DatasetConfig(**dc_[options.dts_name])
        