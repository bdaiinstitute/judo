# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from collections.abc import Iterator as IteratorI
from inspect import Parameter
from typing import Any, Mapping
from typing import Iterator as IteratorT


def is_tuple_t(annotation: Any) -> bool:
    """Check if annotation is of tuple type."""
    return annotation is tuple or (hasattr(annotation, "__origin__") and annotation.__origin__ is tuple)


def is_iterator_t(annotation: Any) -> bool:
    """Check if annotation is of iterable type."""
    return (
        annotation is IteratorT
        or annotation is IteratorI
        or (hasattr(annotation, "__origin__") and annotation.__origin__ is IteratorI)
    )


def positional(parameters: Mapping, index: int) -> Parameter:
    name = list(parameters)[index]
    param = parameters[name]
    if param.kind >= Parameter.KEYWORD_ONLY:
        raise IndexError(f"Parameter {name} is not positional")
    return param
