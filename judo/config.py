# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import importlib
import inspect
import json
import threading
import warnings
from dataclasses import MISSING, dataclass, fields, is_dataclass
from importlib.util import module_from_spec, spec_from_file_location
from typing import Any

import numpy as np

from judo.utils.registration import get_run_id, make_ephemeral_path

_OVERRIDE_REGISTRY: dict[type, dict[str, Any]] = {}

# like in judo/tasks/__init__.py, we use a run ID to create a temporary file for overrides
_run_id = get_run_id()
_OVERRIDES_PATH = make_ephemeral_path("judo_overrides", _run_id)
_override_lock = threading.Lock()


def _load_ephemeral_overrides() -> None:
    """Load any overrides previously registered in this run."""
    if not _OVERRIDES_PATH.is_file():
        return
    try:
        data = json.loads(_OVERRIDES_PATH.read_text())
    except Exception:
        return

    for entry in data:
        cls_mod = entry.get("class_mod")
        cls_qn = entry["class_qn"]

        if cls_mod and cls_mod.startswith("judo."):
            # safe: import the package module (avoids circular file re-exec)
            mod = importlib.import_module(cls_mod)
            cls = getattr(mod, cls_qn)
        else:
            # fallback for __main__ or truly external classes
            spec = spec_from_file_location(f"_judo_override_{cls_qn}", entry["class_src"])
            assert spec is not None, f"Could not load spec for {entry['class_src']}"
            mod = module_from_spec(spec)
            assert spec.loader is not None, f"Loader for {entry['class_src']} is None"
            spec.loader.exec_module(mod)
            cls = getattr(mod, cls_qn)

        _OVERRIDE_REGISTRY[cls] = entry["overrides"]


def _persist_ephemeral_overrides() -> None:
    """Persist the current state of the override registry to disk."""
    with _override_lock:
        serial = []
        for _cls, ov_map in _OVERRIDE_REGISTRY.items():
            src = inspect.getsourcefile(_cls)  # source file of the class
            qn = _cls.__qualname__  # qualified name of the class
            mod = _cls.__module__  # module name of the class
            if src:
                serial.append({"class_src": src, "class_mod": mod, "class_qn": qn, "overrides": ov_map})


_load_ephemeral_overrides()


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

    # persist the overrides to disk so distinct processes can access them
    _persist_ephemeral_overrides()
