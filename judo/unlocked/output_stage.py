# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

from typing import Any

from .annotation import is_tuple_t
from .shmque import SharedMemoryQueue, SharedMemoryQueueView


class OutputStage:
    """Node output stage."""

    def __init__(self, name: str, return_annotation: Any):
        self._name = name
        if return_annotation is None:
            return_annotation = tuple()
        elif is_tuple_t(return_annotation):
            return_annotation = return_annotation.__args__
        else:
            return_annotation = (return_annotation,)

        # the following was meant to satisfy mypy, but it actually made matters worse
        # queue_annotations = tuple((Queue[t] for t in return_annotation))
        # queue_tuple_annotation = tuple[queue_annotations]
        # self._output_queues: queue_tuple_annotation = tuple((Queue[t]() for t in return_annotation))
        self._output_queues: tuple = tuple(
            (SharedMemoryQueue(f"{name}_output_{i}") for i, t in enumerate(return_annotation))
        )
        self._len = len(self._output_queues)

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, channel: int) -> SharedMemoryQueueView:
        if channel < 0 or channel >= self._len:
            raise IndexError("Channel index is out of range")
        return self._output_queues[channel].make_view()

    def open(self) -> None:
        """Open output shared memory queue."""
        for queue in self._output_queues:
            queue.open()

    def close(self) -> None:
        """Close output shared memory queue."""
        for queue in self._output_queues:
            queue.close()

    def push(self, result: tuple) -> None:
        """Push elelemtns to the output stage."""
        for r, q in zip(result, self._output_queues, strict=True):
            q.push(r)
