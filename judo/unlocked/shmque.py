# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
import multiprocessing as mp
import multiprocessing.synchronize
import pickle
import struct
import time
from bisect import bisect
from ctypes import c_size_t, sizeof
from itertools import accumulate, pairwise
from multiprocessing.shared_memory import SharedMemory
from threading import Event, Thread
from typing import Any, Generic, TypeVar

from .const import Const, const

SIZEOF_SIZE_T = sizeof(c_size_t)
SIZE_T_MAX = c_size_t(-1).value
MIN_FRANE_SIZE = 64 * 1024  # 64 kb
T = TypeVar("T")


# Due to a bug in Python 3.10 (https://bugs.python.org/issue39959)
# we need to patch resource tracker for shared memory resource tracking
# TODO (dmitry): remove this hack when we switch to Python 3.13 or later versions
def _patch(name: str, rtype: str):
    '''NoOp to patch resource tracker.'''
    pass
mp.resource_tracker.register = _patch
mp.resource_tracker.unregister = _patch

# inspired by https://github.com/joblib/joblib/issues/1094
class Memory(Generic[T]):
    """Memory class for serialization and deserialization of python objects into and from a memory buffer."""

    def __init__(self, obj: Any):
        self._buffers: list[memoryview] = []
        dump = pickle.dumps(obj, protocol=5, buffer_callback=self._on_buffer)
        self._buffers.append(memoryview(dump))
        self._preamble = b"".join(
            [struct.pack("N", len(self._buffers))] + [struct.pack("N", len(buffer)) for buffer in self._buffers]
        )
        self._buffers = [memoryview(self._preamble), *self._buffers]
        self._size = sum(map(len, self._buffers))

    def _on_buffer(self, buffer: pickle.PickleBuffer) -> None:
        self._buffers.append(buffer.raw())

    def __len__(self) -> int:
        return self._size

    def encode(self, memory: bytearray | memoryview) -> None:
        """Serialize held object into a buffer"""
        if not isinstance(memory, memoryview):
            memory = memoryview(memory)
        assert self._size <= len(memory)
        offset = 0
        for buffer in self._buffers:
            size = len(buffer)
            memory[offset : offset + size] = buffer
            offset += size
        assert offset == self._size

    @classmethod
    def decode(cls, memory: bytearray | memoryview) -> Any:
        """Deserialize from a buffer into an object view (object data may be shared between multiple views)."""
        if not isinstance(memory, memoryview):
            memory = memoryview(memory)
        buffers_num = struct.unpack("N", memory[:SIZEOF_SIZE_T])[0]
        buffers_size: list[int] = [
            struct.unpack("N", memory[(offset + 1) * SIZEOF_SIZE_T : (offset + 2) * SIZEOF_SIZE_T])[0]
            for offset in range(buffers_num)
        ]
        buffers_offset = accumulate(buffers_size, initial=(buffers_num + 1) * SIZEOF_SIZE_T)

        buffers: list[memoryview] = [memory[begin:end] for begin, end in pairwise(buffers_offset)]

        return pickle.loads(buffers[-1], buffers=buffers)


class Frame(Generic[T]):
    """A frame is a fixed size shared memory block that is used by variable size shared memory queue.

    Frame layout
    +--------+--------+-------+--------+------------------------+----------+-------+----------+----------+---------+
    | data_1 | data_2 |  ...  | data_n | //////// free //////// | offset_n |  ...  | offset_3 | offset_2 |    n    |
    +--------+--------+-------+--------+------------------------+----------+-------+----------+----------+---------+
    ^        ^        ^       ^        ^                        ^          ^       ^          ^          ^         ^
    0        |    offset_3    |     free_offset                 |    size - 4(n-1) |      size - 8       |      size
         offset_2          offset_n                        size - 4n           size - 12             size - 4
    """

    def __init__(self, *, name: str, create: bool, size: int | None = None):
        logging.debug("Opening frame%s %s", "" if create else " view", name)
        assert not create or (size is not None), "Cannot specify size for shared memory that has been already created."
        self.name = name
        self._shm: SharedMemory
        self._time = time.perf_counter()
        if size is None:
            self._shm = SharedMemory(name=name, create=create)
        else:
            self._shm = SharedMemory(name=name, create=create, size=size + SIZEOF_SIZE_T)
        self._buffer = self._shm.buf
        self._is_view = not create

        self._len = 0
        self._size = len(self._buffer)
        self._free = self._size - SIZEOF_SIZE_T
        self._used = SIZEOF_SIZE_T
        self._data_offset = [0]
        self._free_offset = 0

        if self._is_view:
            self.free = self.__deleted__  # type: ignore[assignment, method-assign]
            self.used = self.__deleted__  # type: ignore[assignment, method-assign]
            self.push = self.__deleted__  # type: ignore[assignment, method-assign]
            self.push_memory = self.__deleted__  # type: ignore[assignment, method-assign]
            self.data_offset = self._view_data_offset  # type: ignore[method-assign]

    def __deleted__(self, *args: Any, **kwargs: Any) -> None:
        raise RuntimeError("Not implemented for views")

    def __len__(self) -> int:
        return struct.unpack("N", self._buffer[-SIZEOF_SIZE_T:])[0]

    def __del__(self) -> None:
        logging.debug("Closing frame%s %s", " view" if self._is_view else "", self.name)
        self._shm.close()
        if not self._is_view:
            logging.debug("Unlinking frame%s %s", " view" if self._is_view else "", self.name)
            self._shm.unlink()

    def free(self) -> int:
        """Free memory size in bytes."""
        return self._free

    def fits_size(self, size: int) -> bool:
        """Check if an object of a given size fits into the frame."""
        return size + SIZEOF_SIZE_T <= self._free

    def used(self) -> int:
        """Used memory size in bytes."""
        return self._used

    def size(self) -> int:
        """Frame total memory size in bytes."""
        return self._size

    def data_offset(self, index: int) -> int:
        """Memory offset in bytes where an [index] object is stored."""
        if index < 0 or index > len(self):
            raise IndexError("Out of bounds")
        return self._data_offset[index]

    def _view_data_offset(self, index: int) -> int:
        if index < 0 or index >= len(self):
            raise IndexError("Out of bounds")
        if index == 0:
            return 0
        return struct.unpack("N", self._buffer[-(index + 1) * SIZEOF_SIZE_T : -index * SIZEOF_SIZE_T])[0]

    def time(self) -> float:
        """Time at which frame has been created."""
        return self._time

    def __getitem__(self, index: int) -> T:
        if index < 0 or index >= len(self):
            raise IndexError("Out of bounds")
        return Memory.decode(self._buffer[self.data_offset(index) :])

    def push(self, el: T) -> None:
        """Push elelement into the frame."""
        el_mem: Memory = Memory(el)
        self.push_memory(el_mem)

    def push_memory(self, mem: Memory) -> None:
        """Push memory object into the frame.n

        This function saves on serialization time.
        """
        assert not ((self._len == 0) ^ (self._free_offset == 0)), f"What?!?!? {self._len} {self._free_offset}"
        record_size = len(mem) + (0 if self._len == 0 else SIZEOF_SIZE_T)
        if record_size > self._free:
            raise MemoryError(
                f"Frame is out of memory: Record size {record_size} is greater than remaining buffer size {self._free}"
            )
        mem.encode(self._buffer[self._free_offset :])
        if self._len > 0:
            self._buffer[-(self._len + 1) * SIZEOF_SIZE_T : -self._len * SIZEOF_SIZE_T] = struct.pack(
                "N", self._free_offset
            )
            self._data_offset.append(self._free_offset)
        self._len += 1
        self._free_offset += len(mem)

        self._buffer[-SIZEOF_SIZE_T:] = struct.pack("N", self._len)
        self._free -= record_size
        self._used += record_size


class SharedMemoryQueueView(Generic[T]):
    """A view into a shared memory queue.

    A view can read from shared memory frames.
    """

    def __init__(self, ind: int, name: str, *, has_data: mp.synchronize.Condition, shared_frame_count: c_size_t):
        self._ind = ind
        self._name = name
        self._view_name = name + f"_view_{ind}"
        self._has_data = has_data
        self._shared_frame_count = shared_frame_count

        self.index_begin = mp.RawValue(c_size_t, 0)

        self._frames: list[Frame] = []
        self._frame_count = 0
        self._frame_index = [0]
        self._last_frame_length = 0
        self._data_available = Event()
        self._continue_manage = Event()
        # self._data_available = mp.Queue()
        # self._data_available.cancel_join_thread()

        self._view_state = 0

    def open(self) -> None:
        """Open a view.

        A view should be opened before reading the data.
        """
        if self._view_state > 0:
            raise RuntimeError("Shared memory queue view has been opened in this process")
        logging.debug("[%s] Opening", self._view_name)
        self._view_state = 1
        self._managing_thread = Thread(target=self._manage, name=self._view_name + "_manage", daemon=True)
        self._managing_thread.start()

    def close(self) -> None:
        """Close a view.

        A view must be closed in order to clear all resources.
        """
        logging.debug("[%s] Closing %d frames", self._view_name, len(self._frames))
        if self._view_state != 1:
            raise RuntimeError("Cannot close the queue view that has not been opened")
        self._view_state = 2
        logging.debug("[%s] Waiting for manageing thread to finish", self._view_name)
        self._managing_thread.join()
        del self._managing_thread

        self._frame_index = [0]
        logging.debug("[%s] Closing %d frames", self._view_name, len(self._frames))
        del self._frames[:]

        logging.debug("[%s] Setting begin index to max value", self._view_name)
        self.index_begin.value = SIZE_T_MAX
        logging.debug("[%s] Closed", self._view_name)

    def _manage(self) -> None:
        logging.debug("[%s] Starting management loop", self._view_name)
        while self._view_state == 1:
            try:
                if self._sync_frames():
                    self._data_available.set()
                    continue
                with self._has_data:
                    self._has_data.wait(timeout=0.01)
            except BaseException as e:
                logging.error("[%s] Management loop encountered exception:\n%s", self._view_name, str(e))
        logging.debug("[%s] Management loop has finished", self._view_name)

    def _sync_frames(self) -> bool:
        # open frames
        end = self._shared_frame_count.value
        if end != 0:
            if self._frame_count != end:
                logging.debug(
                    "[%s] Adding %d to %d frames", self._view_name, end - self._frame_count, len(self._frames)
                )
                self._frames.extend(
                    [
                        Frame(name=f"{self._name}_frame_{frame_count}", create=False)
                        for frame_count in range(self._frame_count, end)
                    ]
                )
                self._frame_index = list(
                    accumulate((len(frame) for frame in self._frames), initial=self._frame_index[0])
                )
                self._last_frame_length = self._frame_index[-1] - self._frame_index[-2]
                self._frame_count = end
                if len(self) == 0:
                    logging.error("[%s] The queue is empty after new frame has arrived", self._view_name)
                return True
            last_frame_length = len(self._frames[-1])
            if self._last_frame_length != last_frame_length:
                self._frame_index[-1] = self._frame_index[-2] + last_frame_length
                self._last_frame_length = last_frame_length
                if len(self) == 0:
                    logging.error("[%s] The queue is empty after new data has arrived", self._view_name)
                return True

        # close frames
        n = min(bisect(self._frame_index, self.begin) - 1, len(self._frames) - 1)
        if n > 0:
            logging.debug("[%s] Removing %d out of %d frames", self._view_name, n, len(self._frames))
            del self._frames[:n]
            del self._frame_index[:n]

        return False

    def wait(self, timeout: float | None = None) -> bool:
        """Wait for next data."""
        if self._view_state != 1:
            raise RuntimeError("Queue view is not open.")
        if len(self) > 0:
            return True
        # try:
        #     self._data_available.get(timeout=timeout)
        # except queue.Empty:
        #     return Fasle
        if not self._data_available.wait(timeout=timeout):
            return False
        else:
            self._data_available.clear()
        if len(self) == 0:
            logging.error("[%s] The view is empty after data_available event", self._view_name)
            return False

        return True

    @property
    def begin(self) -> int:
        """The first available data index."""
        if self._view_state != 1:
            raise RuntimeError("Queue view is not open.")
        return self.index_begin.value

    @property
    def end(self) -> int:
        """Past last available data index."""
        if self._view_state != 1:
            raise RuntimeError("Queue view is not open.")
        return self._frame_index[-1]

    def __len__(self) -> int:
        return self.end - self.begin

    def __getitem__(self, index: int) -> T | Const[T]:
        if index < self.begin or index >= self.end:
            raise IndexError(f"Index {index} is out of bounds [{self.begin}, {self.end})")
        i = bisect(self._frame_index, index) - 1
        index_in_frame = index - self._frame_index[i]
        assert index_in_frame >= 0, f"index in frame is negative {index_in_frame}"
        frame = self._frames[i]
        assert index_in_frame < len(frame), f"index in frame {index_in_frame} is out of bounds {len(frame)}"
        return const(frame[index_in_frame])

    def top(self) -> T | Const[T]:
        """The top element (FIFO) order."""
        if len(self) < 1:
            raise IndexError("The queue is empty")
        index = self.begin
        i = 1
        while index >= self._frame_index[i]:
            i += 1
        index_in_frame = index - self._frame_index[i - 1]
        assert index_in_frame >= 0, f"index in frame is negative {index_in_frame}"
        frame = self._frames[i - 1]
        assert index_in_frame < len(frame), f"index in frame {index_in_frame} is out of bounds {len(frame)}"
        return const(frame[index_in_frame])

    def pop(self, i: int = 1) -> None:
        """Remove the top element."""
        view_length = len(self)
        if i > view_length:
            raise IndexError(f"Attempting to remove {i} elements from the queue of length {view_length}")
        self.index_begin.value += i
        logging.debug(
            "[%s] Advanced internal counter to %d out of %d",
            self._view_name,
            self.index_begin.value,
            self._frame_index[-1],
        )
        self._data_available.clear()
        logging.debug("[%s] Cleared data_available", self._view_name)


class SharedMemoryQueue(Generic[T]):
    """Shared memory queue is a dynamic FIFO queue.

    Data is stored in a shared memory frames, which can be used in a separate process running on the same host os.
    """

    def __init__(self, name: str, *, min_rate: int = MIN_FRANE_SIZE, fps: int = 1):
        self._name = name
        self._min_rate = min_rate  #  i.e., 1 MB/s the default value
        self._fps = fps  # expected new frames per second (default is 1)
        self._rate_estimate = self._min_rate  # B/s

        self._frame_count = 0
        self._frame_index = [0]
        self._frames: list[Frame] = []

        self._has_data = mp.Condition()

        self._lock = mp.Lock()
        self._view_count = 0
        self._views_index_begin: list[c_size_t] = []
        self._shared_frame_count = mp.RawValue(c_size_t, 0)

        self._queue_state = mp.RawValue(c_size_t, 0)  # 0 -- Initial, 1 -- Open, 2 -- Closed

    # def __getitem__(self, index: int) -> Const | T:
    #     if index < self.begin or index >= self.end:
    #         raise IndexError(f"Index {index} is out of bounds [{self.begin}, {self.end})")
    #     i = bisect(self._frame_index, index) - 1
    #     index_in_frame = index - self._frame_index[i]
    #     assert index_in_frame >= 0, f"index in frame is negative {index_in_frame}"
    #     frame = self._frames[i]
    #     assert index_in_frame < len(frame), f"index in frame {index_in_frame} is out of bounds {len(frame)}"
    #     return const(frame[index_in_frame])
    #
    # def __len__(self) -> int:
    #     return self.end - self.begin

    def __len__(self) -> int:
        return self.end - self.begin

    @property
    def begin(self) -> int:
        """The first available data index."""
        return self._frame_index[0]

    @property
    def end(self) -> int:
        """Past last available data index."""
        return self._frame_index[-1]

    def open(self) -> None:
        """Open a queue.

        A queue should be opened before pushing any data.
        """
        logging.debug("[%s] Opening", self._name)
        with self._lock:
            if self._queue_state.value > 0:
                raise RuntimeError("Shared memory queue can be opened only once")
            self._queue_state.value = 1
        self._managing_thread = Thread(target=self._manage, name=self._name + "_queue_manage", daemon=True)
        self._managing_thread.start()

    def close(self) -> None:
        """Close a queue.

        A queue must be closed in order to clear all resources.
        """
        logging.debug("[%s] Closing with %d frames", self._name, len(self._frames))
        with self._lock:
            if self._queue_state.value != 1:
                raise RuntimeError("Cannot close the queue that has not been opened")
            self._queue_state.value = 2
        self._managing_thread.join()
        del self._managing_thread

        self._frame_index = [0]
        del self._frames[:]

        logging.debug("[%s] Closed", self._name)

    def _manage(self) -> None:
        logging.debug("[%s] Starting management loop", self._name)
        while self._queue_state.value == 1:
            try:
                self.flush()
                time.sleep(0.01)
                # TODO (dmitry): preallocate frames for reducing push latency
                # Currently push lattency is measured ~ 50 ms in the worst case, and ~ 150 us on average
            except BaseException as e:
                logging.error("[%s] Management loop encountered exception:\n%s", self._name, str(e))
        logging.debug("[%s] Finished management loop", self._name)

    def flush(self) -> None:
        """Manually flush unused frames."""
        if len(self._frames) == 0:
            return
        min_begin = min([view.value for view in self._views_index_begin] + [self.end])
        assert min_begin >= self.begin
        # we do not delete the last frame for preventing frequent allocations
        n = min(bisect(self._frame_index, min_begin) - 1, len(self._frames) - 1)
        if n == 0:
            return
        logging.debug("[%s] Flushing %d out of %d frames", self._name, n, len(self._frames))
        del self._frames[:n]
        del self._frame_index[:n]

    def make_view(self) -> SharedMemoryQueueView:
        """Make a queue view."""
        if self._queue_state.value != 0:
            raise RuntimeError("Cannot add views after queue has been opened")
        view: SharedMemoryQueueView = SharedMemoryQueueView(
            ind=self._view_count,
            name=self._name,
            has_data=self._has_data,
            shared_frame_count=self._shared_frame_count,
        )
        self._view_count += 1
        self._views_index_begin.append(view.index_begin)
        return view

    def _add_frame(self, size: int = 0) -> None:
        if self._frames:
            self._rate_estimate = int(
                sum((frame.used() for frame in self._frames)) / (time.perf_counter() - self._frames[0].time())
            )
        logging.debug("[%s] Current rate estimate %d B/s", self._name, self._rate_estimate)
        self._rate_estimate = max(self._fps * self._rate_estimate, self._fps * self._min_rate, size)
        frame: Frame = Frame(name=f"{self._name}_frame_{self._frame_count}", create=True, size=self._rate_estimate)
        self._frames.append(frame)
        self._frame_index.append(self._frame_index[-1])
        self._frame_count += 1
        self._shared_frame_count.value = self._frame_count

    def push(self, el: T) -> None:
        """Push an element into a queue."""
        if self._queue_state.value != 1:
            raise RuntimeError("Queue must be open for pushing elements")
        el_mem: Memory = Memory(el)
        el_size = len(el_mem)
        if not self._frames or not self._frames[-1].fits_size(el_size):
            self._add_frame(el_size)
        self._frames[-1].push(el)
        self._frame_index[-1] += 1
        with self._has_data:
            self._has_data.notify_all()
