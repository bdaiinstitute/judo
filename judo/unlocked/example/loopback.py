# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
import multiprocessing as mp
import signal
from typing import Any, Iterable

# from operator import add
from core.logging_util import setup_logger
from core.unlocked import Node
from core.unlocked.policy import Latest, Lossless
from core.unlocked.schedule import Threaded

setup_logger(log_level=logging.ERROR, include_thread=True)


class StopAll:
    """Terminate all schedulers."""

    def __init__(self, stop_it: Iterable) -> None:
        self._stop = list(stop_it)

    def __call__(self, signum: Any, _: Any) -> None:
        for s in self._stop:
            s.set()


def loop_the_loop(msg: int) -> int:
    """Subscriber printout."""
    return msg + 1


def subscriber(msg: int) -> None:
    logging.error("Received %d", msg)


def main() -> None:
    loop = Node("loop", loop_the_loop, frequency=5000, warmup=[(0,)])  # 5 kHz!
    subs = Node("subs", subscriber, frequency=1)
    loop.input_stage.connect(0, loop.output_stage[0], Lossless())
    subs.input_stage.connect("msg", loop.output_stage[0], Latest())

    schedule1 = Threaded("sch_1", (loop, subs), mp.Event())
    schedule1.start()
    signal.signal(signal.SIGINT, StopAll((schedule1._stop,)))
    schedule1.join()


if __name__ == "__main__":
    main()
