# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from ctypes import c_double, c_uint64
from multiprocessing.sharedctypes import RawValue
from typing import Any

from .shmque import SharedMemoryQueueView


class Lossless:
    """Lossless policy"""

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if the input is ready."""
        return len(view) > 0

    def wait(self, view: SharedMemoryQueueView, timeout: float | None = None) -> bool:
        """Wait for the input."""
        return view.wait(timeout=timeout)

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next element."""
        return view.top()

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        view.pop()


class Latest:
    """Lossless policy"""

    def __init__(self) -> None:
        self._drop = 0
        self._n = RawValue(c_uint64, 0)
        self._min = RawValue(c_uint64, -1)
        self._max = RawValue(c_uint64, 0)
        self._mu = RawValue(c_double, 0.0)
        self._s2 = RawValue(c_double, 0.0)

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if the input is ready."""
        return len(view) > 0

    def wait(self, view: SharedMemoryQueueView, timeout: float | None = None) -> bool:
        """Wait for the input."""
        return view.wait(timeout=timeout)

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next eleent."""
        self._drop = len(view) - 1
        assert self._drop >= 0
        # accumulate dropped message statistics
        self._n.value += 1
        self._min.value = min(self._min.value, self._drop)
        self._max.value = max(self._max.value, self._drop)
        delta = self._drop - self._mu.value
        self._mu.value = self._mu.value + delta / self._n.value
        delta2 = self._drop - self._mu.value
        self._s2.value = self._s2.value + delta * delta2
        # return the latest message
        return view[view.begin + self._drop]

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        view.pop(self._drop + 1)

class Optional:
    """Lossless policy"""

    def __init__(self) -> None:
        self._drop = 0
        self._n = RawValue(c_uint64, 0)
        self._min = RawValue(c_uint64, -1)
        self._max = RawValue(c_uint64, 0)
        self._mu = RawValue(c_double, 0.0)
        self._s2 = RawValue(c_double, 0.0)

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if the input is ready."""
        return True

    def wait(self, view: SharedMemoryQueueView, timeout: float | None = None) -> bool:
        """Wait for the input."""
        return True

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next eleent."""
        self._drop = len(view) - 1
        if self._drop < 0:
            return None
        # accumulate dropped message statistics
        self._n.value += 1
        self._min.value = min(self._min.value, self._drop)
        self._max.value = max(self._max.value, self._drop)
        delta = self._drop - self._mu.value
        self._mu.value = self._mu.value + delta / self._n.value
        delta2 = self._drop - self._mu.value
        self._s2.value = self._s2.value + delta * delta2
        # return the latest message
        return view[view.begin + self._drop]

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        if self._drop < 0:
            return
        view.pop(self._drop + 1)

class KeepLatest:
    """Lossless policy"""

    def __init__(self) -> None:
        self._drop = 0
        self._n = RawValue(c_uint64, 0)
        self._min = RawValue(c_uint64, -1)
        self._max = RawValue(c_uint64, 0)
        self._mu = RawValue(c_double, 0.0)
        self._s2 = RawValue(c_double, 0.0)

    def ready(self, view: SharedMemoryQueueView) -> bool:
        """Check if the input is ready."""
        return len(view) > 0

    def wait(self, view: SharedMemoryQueueView, timeout: float | None = None) -> bool:
        """Wait for the input."""
        return view.wait(timeout=timeout)

    def get(self, view: SharedMemoryQueueView) -> Any:
        """Get the next eleent."""
        self._drop = len(view) - 1
        assert self._drop >= 0
        # accumulate dropped message statistics
        self._n.value += 1
        self._min.value = min(self._min.value, self._drop)
        self._max.value = max(self._max.value, self._drop)
        delta = self._drop - self._mu.value
        self._mu.value = self._mu.value + delta / self._n.value
        delta2 = self._drop - self._mu.value
        self._s2.value = self._s2.value + delta * delta2
        # return the latest message
        return view[view.begin + self._drop]

    def next(self, view: SharedMemoryQueueView) -> None:
        """Advance the input."""
        if self._drop <= 0:
            return
        view.pop(self._drop)
