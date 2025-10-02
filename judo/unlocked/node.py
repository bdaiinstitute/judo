# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
import traceback
from inspect import signature
from time import perf_counter, sleep
from typing import Any, Callable, Iterable

from .input_stage import InputStage
from .output_stage import OutputStage


class NodeStop(Exception):
    """Exception to stop node execution."""


_ADDED_SLEEP_DURATION = 5.41e-5


class Node:
    """Node is a smallest execution unit in the pipeline."""

    def __init__(self, name: str, target: Callable, *, frequency: float | None = None, warmup: Iterable = []):
        self._name = name
        self._target = target
        self._period = None if frequency is None else 1.0 / frequency
        self._warmup = list(warmup)
        self._signature = signature(self._target)
        self._output_stage = OutputStage(self._name, self._signature.return_annotation)
        self._input_stage = InputStage(self._name, self._signature.parameters)
        self._stop: NodeStop | None = None
        self._in_exec = False
        self._count = 0
        self._last_update = perf_counter()

        assert frequency is None or frequency > 0, "Frequency must be positive"
        assert not (len(self.input_stage) == 0 and frequency is None), "Update frequency is required for generators"

    @property
    def name(self) -> str:
        """Check if node has not been terminated."""
        return self._name

    @property
    def live(self) -> bool:
        """Check if node has not been terminated."""
        return self._stop is None

    @property
    def in_exec(self) -> bool:
        """Check if node function is currently running."""
        return self._in_exec

    @property
    def output_stage(self) -> OutputStage:
        """Get output stage."""
        return self._output_stage

    @property
    def input_stage(self) -> InputStage:
        """Get input stage."""
        return self._input_stage

    @property
    def ready(self) -> bool:
        """Get input stage."""
        if self._period is not None and perf_counter() - self._last_update < self._period:
            return False
        return self._input_stage.ready

    def wait(self, timeout: int | None = None) -> bool:
        """Wait for the input data to arrive."""
        if self._period is None:
            return self._input_stage.wait(timeout=timeout)
        time_left = self._period + self._last_update - _ADDED_SLEEP_DURATION - perf_counter()
        if timeout is not None and time_left > timeout - _ADDED_SLEEP_DURATION:
            sleep(timeout - _ADDED_SLEEP_DURATION)
            return False
        if time_left > 0:
            sleep(time_left)
        if not self.live:
            return False
        return self._input_stage.wait(timeout=0.5)

    def open(self) -> None:
        """Open node shared memory communications."""
        self._output_stage.open()
        for w in self._warmup:
            self._output_stage.push(w)
        self._input_stage.open()

    def close(self) -> None:
        """Close node shared memory communications."""
        self._stop = NodeStop("Closed nominally.")
        self._input_stage.close()
        self._output_stage.close()

    def exec(self) -> None:
        """Run the node function."""
        if self._period is not None:
            time_passed = perf_counter() - self._last_update
            if time_passed <= 1.5 * self._period:
                self._last_update += self._period
            else:
                delay_percent = int(100 * (time_passed / self._period - 1.0))
                # logging.warning("Node %s execution is delayed by %d %%", self._name, delay_percent)
                self._last_update += time_passed
        assert self.live, "The node has been stopped."
        self._in_exec = True
        assert self._input_stage.ready, "Input stage is not ready."
        args, kwargs = self._input_stage.input_args()
        try:
            result = self._target(*args, **kwargs)
        except BaseException as e:
            self._set_error(e)
            return

        self._push_result(result)

    def _push_result(self, result: Any) -> None:
        if result is None:
            result = tuple()
        elif not isinstance(result, tuple):
            result = (result,)
        self._input_stage.next()
        self._output_stage.push(result)
        self._in_exec = False
        self._count += 1

    def _set_error(self, error: Any) -> None:
        assert error is not None
        if isinstance(error, NodeStop):
            self._stop = error
            logging.info("Node %s stopped with %s", self._name, error)
        else:
            logging.error("Node %s failed with %s\n === traceback ===\n%s =================", self._name, error, traceback.format_exc())
        self._in_exec = False
