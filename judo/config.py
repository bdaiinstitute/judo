# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from dataclasses import MISSING, dataclass, fields, is_dataclass
from typing import Any

import numpy as np

_OVERRIDE_REGISTRY: dict[type, dict[str, Any]] = {}


@dataclass
class OverridableConfig:
    """A class that provides an interface to switch between its field values depending on an override key."""

    def __post_init__(self) -> None:
        """Initialize the override key to 'default'."""
        if _OVERRIDE_REGISTRY.get(self.__class__, None) is None:
            _OVERRIDE_REGISTRY[self.__class__] = {}

    def set_override(self, key: str, reset_to_defaults: bool = True) -> None:
        """Set the overridden values for the config based on the override registry.

        Args:
            key: The key to use for the override.
            reset_to_defaults: If True, reset the values to their defaults if no override is found. This is useful for
                when you switch from different non-default overrides to other non-default overrides.
        """
        class_specific_overrides = _OVERRIDE_REGISTRY.get(self.__class__, {})
        active_key_overrides = class_specific_overrides.get(key, {})

        for f in fields(self):
            override_value = active_key_overrides.get(f.name)

            if override_value is not None:
                current_value = getattr(self, f.name, MISSING)
                if current_value != override_value:
                    setattr(self, f.name, override_value)
            elif reset_to_defaults:
                default_value_to_set = MISSING

                # handle default and default_factory
                if f.default is not MISSING:
                    default_value_to_set = f.default
                elif f.default_factory is not MISSING:
                    default_value_to_set = f.default_factory()

                # set default value if it exists
                if default_value_to_set is not MISSING:
                    current_value = getattr(self, f.name, MISSING)  # Get current value
                    if isinstance(current_value, np.ndarray) and isinstance(default_value_to_set, np.ndarray):
                        if not np.array_equal(current_value, default_value_to_set):
                            setattr(self, f.name, default_value_to_set)
                    elif current_value != default_value_to_set:
                        setattr(self, f.name, default_value_to_set)
                else:
                    warnings.warn(
                        f"Field '{f.name}' has no default value to reset to and no override for key '{key}'. "
                        "Its current value remains unchanged.",
                        UserWarning,
                        stacklevel=2,
                    )


def set_config_overrides(
    override_key: str,
    cls: type,
    field_override_values: dict[str, Any],
) -> None:
    """Modify the override registry to include an override key and value.

    Can also be used to choose new override values for an existing key.

    Args:
        override_key: The key to use for the override.
        cls: The class to modify.
        field_override_values: A dictionary of field names and their corresponding override values.
    """
    if not is_dataclass(cls):
        raise TypeError(f"Provided class {cls.__name__} is not a dataclass.")
    if _OVERRIDE_REGISTRY.get(cls, None) is None:
        _OVERRIDE_REGISTRY[cls] = {override_key: {}}
    if _OVERRIDE_REGISTRY[cls].get(override_key, None) is None:
        _OVERRIDE_REGISTRY[cls][override_key] = {}

    cls_field_names = {f.name for f in fields(cls)}
    for field_name, override_value in field_override_values.items():
        if field_name in cls_field_names:
            _OVERRIDE_REGISTRY[cls][override_key][field_name] = override_value
        else:
            warnings.warn(
                f"Field '{field_name}' not found in class '{cls.__name__}'. "
                f"No override value added for this field under key '{override_key}'.",
                UserWarning,
                stacklevel=2,
            )
