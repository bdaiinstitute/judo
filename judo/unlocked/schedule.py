# Copyright (c) 2025 Boston Dynamics AI Institute LLC. All rights reserved.

import logging
import multiprocessing as mp
import multiprocessing.synchronize
import signal
from threading import Thread
from time import sleep
from typing import Iterable

from .node import Node


def _node(node: Node) -> None:
    while node.live:
        if node.wait():
            node.exec()


class Threaded:
    """Threaded scheduler."""

    def __init__(self, name: str, nodes: Iterable[Node], stop: mp.synchronize.Event):
        self._name = name
        self._nodes = list(nodes)
        self._proc = mp.Process(target=self.exec, name=name)
        self._stop = stop

    def start(self) -> None:
        """Start scheduler."""
        self._proc.start()

    def join(self) -> None:
        """Join scheduler."""
        self._proc.join()

    def exec(self) -> None:
        """Run execution loop."""
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        self.spin()

    def spin(self) -> None:
        logging.debug("Scheduler %s has started", self._name)

        for node in self._nodes:
            node.open()

        self._start_node_threads()

        while not self._stop.is_set():
            sleep(0.1)

        for node in self._nodes:
            node.close()

        logging.debug("Scheduler %s has stopped", self._name)

    def _start_node_threads(self) -> None:
        self.node_threads = [
            Thread(
                target=_node,
                args=(node,),
                name=node.name,
                daemon=True,
            )
            for node_ind, node in enumerate(self._nodes)
        ]
        for t in self.node_threads:
            t.start()
