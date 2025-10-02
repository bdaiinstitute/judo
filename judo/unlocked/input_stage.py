# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from inspect import Parameter
from typing import Any, Mapping
from typing import Any as Policy

from .shmque import SharedMemoryQueueView


class InputStage:
    """Node output stage."""

    def __init__(self, name: str, parameters: Mapping):
        self._name = name
        self._parameters = parameters
        self._inputs: dict[str | int, tuple[SharedMemoryQueueView, Policy]] = {}
        self._arg_len = 0
        self._arg_inputs: list[tuple[SharedMemoryQueueView, Policy]] = []
        self._kwarg_inputs: list[tuple[str, SharedMemoryQueueView, Policy]] = []

    def __len__(self) -> int:
        """Number of inputs."""
        return len(self._parameters)

    @property
    def ready(self) -> bool:
        """Check if data is available."""
        return all((policy.ready(view) for view, policy in self._inputs.values()))

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for the data."""
        for view, policy in self._inputs.values():
            if not policy.wait(view, timeout=timeout):
                return False
        return True

    def open(self) -> None:
        """Open input shared memory views."""
        self._assert_contiguous_positional_arguments()
        self._arg_inputs = [self._inputs[arg] for arg in range(self._arg_len)]
        self._kwarg_inputs = [(key, *value) for key, value in self._inputs.items() if not isinstance(key, int)]
        for view, _ in self._inputs.values():
            view.open()

    def close(self) -> None:
        """Close input shared memory views."""
        for view, _ in self._inputs.values():
            view.close()

    def connect(self, arg: int | str, queue_view: SharedMemoryQueueView, policy: Policy) -> None:
        """Connect a view to a positional of keyword input argument."""
        if (
            not isinstance(arg, int)
            and arg in self._parameters
            and self._parameters[arg].kind == Parameter.POSITIONAL_OR_KEYWORD
        ):
            arg = list(self._parameters.keys()).index(arg)

        if arg in self._inputs:
            raise ValueError("Input stage cannot subscribe to multiple output stages.")

        self._inputs[arg] = (queue_view, policy)

    def input_args(self) -> tuple[list[Any], dict[str, Any]]:
        """Get input arguments from the views."""
        args = [policy.get(view) for view, policy in self._arg_inputs]
        kwargs = {name: policy.get(view) for name, view, policy in self._kwarg_inputs}
        return args, kwargs

    def next(self) -> None:
        """Pop data from the views."""
        for view, policy in self._inputs.values():
            policy.next(view)

    def _assert_contiguous_positional_arguments(self) -> None:
        if len(self._inputs) == 0:
            return

        is_min_zero = False
        for arg in self._inputs.keys():
            if not isinstance(arg, int):
                continue
            if arg == 0:
                is_min_zero = True
            self._arg_len = max(self._arg_len, arg + 1)

        assert is_min_zero, "Missing the first positional argument"

        for arg in range(self._arg_len):
            assert arg in self._inputs, f"Missing the {arg}th positional argument"
