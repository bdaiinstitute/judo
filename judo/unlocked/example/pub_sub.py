# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
import multiprocessing as mp
import signal
from typing import Any, Iterable

# from operator import add
from core.logging_util import setup_logger
from core.unlocked import Node
from core.unlocked.policy import Latest
from core.unlocked.schedule import Threaded

setup_logger(log_level=logging.ERROR)


class StopAll:
    """Terminate all schedulers."""

    def __init__(self, stop_it: Iterable) -> None:
        self._stop = list(stop_it)

    def __call__(self, signum: Any, _: Any) -> None:
        for s in self._stop:
            s.set()


class Publisher:
    """Publishing generator."""

    def __init__(self) -> None:
        self._counter = 0

    def __call__(self) -> int:
        self._counter += 1
        # logging.info("Sending %d", self._counter)
        return self._counter


def subscriber(msg: int) -> None:
    """Subscriber printout."""
    logging.error("Received %d", msg)


def main() -> None:
    pub = Node("pub", Publisher(), frequency=10000)  # 10 kHz!
    sub = Node("sub", subscriber, frequency=1)
    sub.input_stage.connect("msg", pub.output_stage[0], Latest())

    stop = mp.Event()
    schedule1 = Threaded(
        "sch_1",
        (
            pub,
            sub,
        ),
        stop,
    )
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((stop,)))
    schedule1.join()


if __name__ == "__main__":
    main()
