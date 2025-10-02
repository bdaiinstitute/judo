# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from contextlib import nullcontext
from ctypes import c_double, c_uint64
from math import sqrt
from multiprocessing import Semaphore
from multiprocessing.sharedctypes import RawValue
from time import perf_counter_ns
from typing import Any, Generic, TypeVar

T = TypeVar("T")


def _f(nanoseconds: float) -> str:
    if nanoseconds >= 1.0e9:
        return f"{(nanoseconds*1.e-9):6.2f}  s"
    if nanoseconds >= 1.0e6:
        return f"{(nanoseconds*1.e-6):6.2f} ms"
    if nanoseconds >= 1.0e3:
        return f"{(nanoseconds*1.e-3):6.2f} us"
    return f"{(nanoseconds):6.2f} ns"


class SharedTimeStat:
    """Accumulate time statistics across multiple processes."""

    def __init__(self, name: str, *, lock: bool = True):
        self._mutex: Mutex | nullcontext = Mutex(name + "_mutex") if lock else nullcontext()
        self._n = RawValue(c_uint64, 0)
        self._min = RawValue(c_uint64, -1)
        self._max = RawValue(c_uint64, 0)
        self._mu = RawValue(c_double, 0.0)
        self._s2 = RawValue(c_double, 0.0)
        self._tic = RawValue(c_uint64, 0)

    def __enter__(self) -> "SharedTimeStat":
        self.tic()
        return self

    def __exit__(self, *args: Any) -> None:
        self.toc()

    def __str__(self) -> str:
        with self._mutex:
            n = self._n.value
            if n == 0:
                return "================================= No Data ================================="
            return (
                f"n = {n}, "
                f"min = {_f(self._min.value)}, "
                f"max = {_f(self._max.value)}, "
                f"avg = {_f(self._mu.value)}, "
                f"std = {_f(sqrt(self._s2.value / n))}"
            )

    def clear(self) -> None:
        """Clear accumulated statistics."""
        with self._mutex:
            self._n.value = 0
            self._min.value = -1
            self._max.value = 0
            self._mu.value = 0
            self._s2.value = 0
            self._tic.value = 0

    def tic(self, now: int | None = None) -> int:
        """Start time tracking."""
        if now is None:
            now = perf_counter_ns()
        with self._mutex:
            self._tic.value = now
        return now

    def toc(self, now: int | None = None) -> None:
        """Stop time tracking and accumulated staticstics."""
        if now is None:
            now = perf_counter_ns()
        with self._mutex:
            self._n.value += 1
            x = now - self._tic.value
            self._min.value = min(self._min.value, x)
            self._max.value = max(self._max.value, x)
            delta = x - self._mu.value
            self._mu.value = self._mu.value + delta / self._n.value
            delta2 = x - self._mu.value
            self._s2.value = self._s2.value + delta * delta2

    @property
    def n(self) -> int:
        """Number of records."""
        with self._mutex:
            return self._n.value

    @property
    def mu(self) -> float:
        """Avarage time in seconds."""
        with self._mutex:
            return self._mu.value * 1.0e-9

    @property
    def sigma(self) -> float:
        """Time standard deviation in seconds."""
        with self._mutex:
            n = self._n.value
            if n == 0:
                return float("nan")
            return sqrt(self._s2.value / self._n.value) * 1.0e-9

    @property
    def min(self) -> float:
        """Minimum time in seconds."""
        with self._mutex:
            return self._min.value * 1.0e-9

    @property
    def max(self) -> float:
        """Maximum time in seconds."""
        with self._mutex:
            return self._max.value * 1.0e-9


# We are preparing for using POSIX Semaphores to be able to synch with C++/Rust/Zig code.
# (POSIX Semaphores)[https://github.com/osvenskan/posix_ipc/blob/develop/USAGE.md#the-semaphore-class]
# We also will create custom Test Semaphores that will simulate random arrival times.
class NamedSemaphore:
    """Posix semaphore."""

    def __init__(self, name: str, value: int = 1):
        self.__name__ = name
        self._semaphore = Semaphore(value)

    def acquire(self) -> None:
        """Acquire semaphore."""
        self._semaphore.acquire()

    def release(self, n: int = 1) -> None:
        """Release semaphore."""
        if n == 1:  # it's a bit faster to do if then a for loop of 1
            self._semaphore.release()
        else:
            for _ in range(n):
                self._semaphore.release()


class UseGenericSemaphore(Generic[T]):
    """Templated semaphore."""

    __SemaphoreT__: type = NamedSemaphore

    def __class_getitem__(cls, SemaphoreT: type) -> type:
        UseSemaphore = cls
        UseSemaphore.__SemaphoreT__ = SemaphoreT
        return UseSemaphore


class Mutex(UseGenericSemaphore):
    """Mutex."""

    def __init__(self, name: str):
        self._name = name
        self._s = self.__SemaphoreT__(name + "_s", 1)

    def __enter__(self) -> None:
        self._s.acquire()

    def __exit__(self, *args: Any) -> None:
        self._s.release()

    def acquire(self) -> None:
        """Manually acquire mutex."""
        self._s.acquire()

    def release(self) -> None:
        """Manually release mutex."""
        self._s.release()


class Event(UseGenericSemaphore):
    """Event syncronizes a calling process with one and only one of waiting processes."""

    def __init__(self, name: str):
        self._name = name
        self._s1 = self.__SemaphoreT__(name + "_s1", 0)
        self._s2 = self.__SemaphoreT__(name + "_s2", 0)

    def __enter__(self) -> None:
        self._s1.acquire()

    def __exit__(self, *args: Any) -> None:
        self._s2.release()

    def notify(self) -> None:
        """Notify one waiting process and wait for it to wake up."""
        self._s1.release()
        self._s2.acquire()


class BroadcastEvent(UseGenericSemaphore):
    """Broadcast event syncronizes a calling process with all (any number) of waiting processes."""

    def __init__(self, name: str):
        self._name = name
        self._counter = RawValue(c_uint64, 0)
        self._mutex = Mutex[self.__SemaphoreT__](name + "_mutex")  # type: ignore[misc]
        self._t = Mutex[self.__SemaphoreT__](name + "_turn")  # type: ignore[misc]

        self._s1 = self.__SemaphoreT__(name + "_s1", 0)
        self._s2 = self.__SemaphoreT__(name + "_s2", 1)
        self._s3 = self.__SemaphoreT__(name + "_s3", 0)

    def __enter__(self) -> None:
        # Block until notifier is done
        with self._t:
            # Count number of listeners
            with self._mutex:
                self._counter.value += 1

        # Wait for the notice
        self._s1.acquire()

    def __exit__(self, *args: Any) -> None:
        # Release listeners
        with self._mutex:
            self._counter.value -= 1
            if self._counter.value == 0:
                self._s2.release()
                self._s3.release()

        self._s2.acquire()
        self._s2.release()

    def notify(self) -> None:
        """Notify all waiting process and wait for all of them to wake up."""
        # Notifier block
        with self._t:
            with self._mutex:
                n = self._counter.value
            # if no listeners are waiting---leave
            if n == 0:
                return
            # notify
            self._s2.acquire()
            self._s1.release(n)
            self._s3.acquire()


class Barrier(UseGenericSemaphore):
    """Barrier syncronizes exactly n processes."""

    def __init__(self, name: str, n: int):
        self._name = name
        self._n = n
        self._counter = RawValue(c_uint64, 0)
        self._mutex = Mutex[self.__SemaphoreT__](name + "_mutex")  # type: ignore[misc]
        self._s1 = self.__SemaphoreT__(name + "_s1", 0)
        self._s2 = self.__SemaphoreT__(name + "_s2", 1)

    def __enter__(self) -> None:
        with self._mutex:
            self._counter.value += 1
            if self._counter.value == self._n:
                self._s2.acquire()
                self._s1.release(self._n + 1)

        self._s1.acquire()

    def __exit__(self, *args: Any) -> None:
        with self._mutex:
            self._counter.value -= 1
            if self._counter.value == 0:
                self._s1.acquire()
                self._s2.release(self._n + 1)

        self._s2.acquire()

    # A slower, but perhaps a safer solution, in which threads are unlocked one-by-one.
    # def __enter__(self) -> None:
    #     with self._mutex:
    #         self._counter.value += 1
    #         if self._counter.value == self._n:
    #             self._s2.acquire()
    #             self._s1.release()
    #
    #     self._s1.acquire()
    #     self._s1.release()
    #
    # def __exit__(self, *args) -> None:
    #     with self._mutex:
    #         self._counter.value -= 1
    #         if self._counter.value == 0:
    #             self._s1.acquire()
    #             self._s2.release()
    #
    #     self._s2.acquire()
    #     self._s2.release()
