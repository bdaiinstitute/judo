# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import warnings
from dataclasses import dataclass, field
from typing import Any, Generator

import pytest

from judo.config import _OVERRIDE_REGISTRY, OverridableConfig, set_config_overrides

# ##### #
# SETUP #
# ##### #


@pytest.fixture(autouse=True)
def clear_override_registry() -> Generator:
    """Clears the global _OVERRIDE_REGISTRY before each test."""
    _OVERRIDE_REGISTRY.clear()
    yield  # allow test to run


@dataclass
class SimpleConfig(OverridableConfig):
    """A simple configuration class with default values."""

    param_a: int = 10
    param_b: str = "default_b"
    param_c: bool = True


@dataclass
class ConfigWithRequiredValues(OverridableConfig):
    """A configuration class with required parameters and no default values."""

    required_param: str
    another_required: int
    optional_param: float = 0.5


@dataclass
class ConfigWithDefaultFactory(OverridableConfig):
    """A configuration class with default factory values."""

    my_list: list[int] = field(default_factory=list)
    my_dict: dict[str, Any] = field(default_factory=dict)
    fixed_val: int = 100


class NotADataclass:
    """A class that is not a dataclass."""


@pytest.fixture
def simple_config() -> SimpleConfig:
    """Fixture for creating a SimpleConfig instance."""
    return SimpleConfig()


@pytest.fixture
def config_no_defaults() -> ConfigWithRequiredValues:
    """Fixture for creating a ConfigWithRequiredValues instance."""
    # initialize required fields for tests that don't immediately override them
    return ConfigWithRequiredValues(required_param="initial_req", another_required=0)


@pytest.fixture
def config_default_factory() -> ConfigWithDefaultFactory:
    """Fixture for creating a ConfigWithDefaultFactory instance."""
    return ConfigWithDefaultFactory()


# ##### #
# TESTS #
# ##### #


def test_set_config_overrides_new_class_and_key() -> None:
    """Test setting config overrides for a new class and key."""
    set_config_overrides("test_env", SimpleConfig, {"param_a": 100, "param_b": "env_b"})
    assert SimpleConfig in _OVERRIDE_REGISTRY
    assert "test_env" in _OVERRIDE_REGISTRY[SimpleConfig]
    assert _OVERRIDE_REGISTRY[SimpleConfig]["test_env"]["param_a"] == 100
    assert _OVERRIDE_REGISTRY[SimpleConfig]["test_env"]["param_b"] == "env_b"


def test_set_config_overrides_existing_class_new_key() -> None:
    """Test setting config overrides for an existing class and a new key."""
    set_config_overrides("prod", SimpleConfig, {"param_a": 200})
    set_config_overrides("staging", SimpleConfig, {"param_b": "staging_b"})

    assert "prod" in _OVERRIDE_REGISTRY[SimpleConfig]
    assert _OVERRIDE_REGISTRY[SimpleConfig]["prod"]["param_a"] == 200
    assert "staging" in _OVERRIDE_REGISTRY[SimpleConfig]
    assert _OVERRIDE_REGISTRY[SimpleConfig]["staging"]["param_b"] == "staging_b"


def test_set_config_overrides_update_existing_key_value() -> None:
    """Test updating an existing key's value in the override registry."""
    set_config_overrides("test_env", SimpleConfig, {"param_a": 100})
    set_config_overrides("test_env", SimpleConfig, {"param_a": 150, "param_b": "updated_b"})

    assert _OVERRIDE_REGISTRY[SimpleConfig]["test_env"]["param_a"] == 150
    assert _OVERRIDE_REGISTRY[SimpleConfig]["test_env"]["param_b"] == "updated_b"


def test_set_config_overrides_non_existent_field_issues_warning() -> None:
    """Test that setting a non-existent field issues a warning and does not affect the registry."""
    with pytest.warns(UserWarning, match="Field 'non_existent' not found in class 'SimpleConfig'"):
        set_config_overrides("test_env", SimpleConfig, {"non_existent": 999, "param_a": 50})

    # ensure the valid field was set and the invalid one was not
    assert SimpleConfig in _OVERRIDE_REGISTRY
    assert "test_env" in _OVERRIDE_REGISTRY[SimpleConfig]
    assert "non_existent" not in _OVERRIDE_REGISTRY[SimpleConfig]["test_env"]
    assert _OVERRIDE_REGISTRY[SimpleConfig]["test_env"]["param_a"] == 50


def test_set_config_overrides_not_a_dataclass_raises_typeerror() -> None:
    """Test that providing a non-dataclass raises a TypeError."""
    with pytest.raises(TypeError, match="Provided class NotADataclass is not a dataclass."):
        set_config_overrides("test_env", NotADataclass, {"some_field": 1})


def test_set_config_overrides_empty_field_values_is_allowed() -> None:
    """Test that setting empty field values is allowed."""
    set_config_overrides("empty_env", SimpleConfig, {})
    assert SimpleConfig in _OVERRIDE_REGISTRY
    assert "empty_env" in _OVERRIDE_REGISTRY[SimpleConfig]
    assert _OVERRIDE_REGISTRY[SimpleConfig]["empty_env"] == {}


def test_overridable_config_initialization(simple_config: SimpleConfig) -> None:
    """Test that OverridableConfig initializes correctly."""
    assert simple_config.param_a == 10
    assert simple_config.param_b == "default_b"
    assert simple_config.__class__ in _OVERRIDE_REGISTRY


def test_post_init_does_not_overwrite_existing_registry_for_class() -> None:
    """Test that __post_init__ does not overwrite existing registry for the class."""
    _OVERRIDE_REGISTRY[SimpleConfig] = {"preset_key": {"param_a": 123}}
    _cfg = SimpleConfig()  # __post_init__ is called
    assert "preset_key" in _OVERRIDE_REGISTRY[SimpleConfig]
    assert _OVERRIDE_REGISTRY[SimpleConfig]["preset_key"]["param_a"] == 123


def test_set_override_applies_overrides_and_resets_missing_to_defaults(simple_config: SimpleConfig) -> None:
    """Test that set_override applies overrides and resets missing fields to defaults."""
    set_config_overrides("prod", SimpleConfig, {"param_a": 100, "param_c": False})
    simple_config.param_b = "changed_b"  # manually change a field not in override

    simple_config.set_override("prod")

    assert simple_config.param_a == 100  # overridden
    assert simple_config.param_c is False  # overridden
    assert simple_config.param_b == "default_b"  # reset to default


def test_set_override_applies_overrides_no_reset_to_defaults(simple_config: SimpleConfig) -> None:
    """Test that set_override applies overrides without resetting to defaults."""
    set_config_overrides("prod", SimpleConfig, {"param_a": 100})

    # these should persist
    simple_config.param_b = "custom_value"
    simple_config.param_c = False

    simple_config.set_override("prod", reset_to_defaults=False)

    assert simple_config.param_a == 100  # overridden
    assert simple_config.param_b == "custom_value"  # not reset
    assert simple_config.param_c is False  # not reset


def test_set_override_key_not_in_registry_resets_to_defaults(simple_config: SimpleConfig) -> None:
    """Test that set_override for a key not in the registry resets to defaults."""
    simple_config.param_a = 50
    simple_config.param_b = "temp_b"

    simple_config.set_override("non_existent_key")

    assert simple_config.param_a == 10  # reset
    assert simple_config.param_b == "default_b"  # reset


def test_set_override_key_not_in_registry_no_reset(simple_config: SimpleConfig) -> None:
    """Test that set_override for a key not in the registry does not reset to defaults."""
    simple_config.param_a = 999
    simple_config.param_b = "xyz"

    simple_config.set_override("non_existent_key_no_reset", reset_to_defaults=False)

    assert simple_config.param_a == 999  # unchanged
    assert simple_config.param_b == "xyz"  # unchanged


def test_set_override_field_has_no_default_and_no_override_warns_and_unchanged(
    config_no_defaults: ConfigWithRequiredValues,
) -> None:
    """Test that setting an override with no default and no override issues a warning and leaves it unchanged."""
    set_config_overrides("env1", ConfigWithRequiredValues, {"optional_param": 1.5})

    # required_param and another_required have no defaults and are not in this override set
    config_no_defaults.required_param = "specific_value"
    config_no_defaults.another_required = 77

    with pytest.warns(UserWarning) as warnings_info:
        config_no_defaults.set_override("env1")  # reset_to_defaults=True

    assert config_no_defaults.optional_param == 1.5  # overridden
    assert config_no_defaults.required_param == "specific_value"  # unchanged
    assert config_no_defaults.another_required == 77  # unchanged

    # check warnings content
    warn_messages = [str(w.message) for w in warnings_info]
    assert any(
        "Field 'required_param' has no default value to reset to and no override for key 'env1'" in m
        for m in warn_messages
    )
    assert any(
        "Field 'another_required' has no default value to reset to and no override for key 'env1'" in m
        for m in warn_messages
    )


def test_set_override_field_has_no_default_no_override_no_reset_unchanged(
    config_no_defaults: ConfigWithRequiredValues,
) -> None:
    """Test that setting an override with no default and no override doesn't change field if reset_to_defaults=False."""
    set_config_overrides("env1", ConfigWithRequiredValues, {"optional_param": 2.5})
    config_no_defaults.required_param = "val_req"
    config_no_defaults.another_required = 88

    with warnings.catch_warnings(record=True) as w:  # should not warn
        config_no_defaults.set_override("env1", reset_to_defaults=False)
        assert len(w) == 0

    assert config_no_defaults.optional_param == 2.5  # overridden
    assert config_no_defaults.required_param == "val_req"  # unchanged
    assert config_no_defaults.another_required == 88  # unchanged


def test_set_override_switching_between_keys(simple_config: SimpleConfig) -> None:
    """Test switching between different override keys."""
    set_config_overrides("dev", SimpleConfig, {"param_a": 1, "param_b": "dev_b"})
    set_config_overrides("prod", SimpleConfig, {"param_a": 1000, "param_b": "prod_b", "param_c": False})

    simple_config.set_override("dev")
    assert simple_config.param_a == 1
    assert simple_config.param_b == "dev_b"
    assert simple_config.param_c is True  # teset to default

    simple_config.set_override("prod")
    assert simple_config.param_a == 1000
    assert simple_config.param_b == "prod_b"
    assert simple_config.param_c is False  # overridden

    simple_config.set_override("dev")  # switch back
    assert simple_config.param_a == 1
    assert simple_config.param_b == "dev_b"
    assert simple_config.param_c is True  # reset to default again


def test_overridable_config_with_default_factory(config_default_factory: ConfigWithDefaultFactory) -> None:
    """Test that default_factory fields are reset to new instances when overridden."""
    cfg = config_default_factory
    assert cfg.my_list == []
    assert cfg.my_dict == {}
    assert cfg.fixed_val == 100

    cfg.my_list.append(1)
    cfg.my_dict["key"] = "value"

    override_list = [10, 20]
    set_config_overrides("env_df", ConfigWithDefaultFactory, {"my_list": override_list, "fixed_val": 200})

    cfg.set_override("env_df")
    assert cfg.my_list is override_list  # check if it's the same object for mutable types
    assert cfg.my_dict == {}  # reset to new default_factory result
    assert cfg.fixed_val == 200

    # ensure default_factory creates new objects on reset
    cfg2 = ConfigWithDefaultFactory()
    cfg2.my_list.append(99)
    cfg2.set_override("env_df")  # my_list gets override_list
    assert cfg.my_list is override_list
    assert cfg2.my_list is override_list  # Both point to the same override list

    # if we reset to a key without my_list override, it should get a new list
    set_config_overrides("env_df_no_list", ConfigWithDefaultFactory, {"fixed_val": 300})
    cfg.set_override("env_df_no_list")
    assert cfg.my_list == []  # new list from factory
    assert cfg.my_list is not override_list
    assert cfg.fixed_val == 300


def test_multiple_instances_independent_state_after_override() -> None:
    """Test that multiple instances of the same class can have independent states after applying overrides."""
    set_config_overrides("env_multi", SimpleConfig, {"param_a": 555})

    instance1 = SimpleConfig()  # param_a = 10
    instance2 = SimpleConfig()  # param_a = 10

    instance1.set_override("env_multi")
    assert instance1.param_a == 555
    assert instance2.param_a == 10  # instance2 not affected yet

    instance2.set_override("env_multi")
    assert instance2.param_a == 555

    # change instance1's state *after* override, without changing override key
    instance1.param_a = 666
    assert instance1.param_a == 666
    assert instance2.param_a == 555  # instance2 remains at its last set override value

    # if global override definition changes, re-applying set_override picks it up
    set_config_overrides("env_multi", SimpleConfig, {"param_a": 777})
    instance1.set_override("env_multi")  # re-apply
    instance2.set_override("env_multi")  # re-apply
    assert instance1.param_a == 777
    assert instance2.param_a == 777
