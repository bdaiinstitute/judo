# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
import multiprocessing as mp
import signal
from typing import Any, Iterable

# from operator import add
from core.logging_util import setup_logger
from core.unlocked import Node
from core.unlocked.policy import Lossless
from core.unlocked.schedule import Threaded

setup_logger(log_level=logging.INFO)


class StopAll:
    """Terminate all schedulers."""

    def __init__(self, stop_it: Iterable) -> None:
        self._stop = list(stop_it)

    def __call__(self, signum: Any, _: Any) -> None:
        for s in self._stop:
            s.set()


def fibonacci(a: int, b: int) -> tuple[int, int]:
    """Subscriber printout."""
    next = a + b
    logging.info("Next fibonacci number is %d", next)
    return b, next


def main() -> None:
    f_node = Node("fibonacci", fibonacci, frequency=10, warmup=[(0, 1)])
    f_node.input_stage.connect(0, f_node.output_stage[0], Lossless())
    f_node.input_stage.connect(1, f_node.output_stage[1], Lossless())

    stop = mp.Event()
    schedule1 = Threaded("sch_1", (f_node,), stop)
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((stop,)))
    schedule1.join()


if __name__ == "__main__":
    main()
