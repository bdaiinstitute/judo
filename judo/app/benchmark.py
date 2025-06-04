# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import sys
from collections import defaultdict

import numpy as np
import pyarrow as pa
from dora_utils.node import DoraNode, on_event
from rich.console import Console
from rich.table import Table

from judo.optimizers import get_registered_optimizers
from judo.tasks import get_registered_tasks


class BenchmarkerNode(DoraNode):
    """A node that performs benchmarking of task/optimizer pairs."""

    def __init__(self, node_id: str = "benchmarker", max_workers: int | None = None, num_samples: int = 100) -> None:
        """Initialize the benchmarker node."""
        super().__init__(node_id=node_id, max_workers=max_workers)

        # storing all available tasks and optimizers
        self.available_optimizers = get_registered_optimizers()
        self.available_tasks = get_registered_tasks()
        self.opt_keys = list(self.available_optimizers.keys())
        self.task_keys = list(self.available_tasks.keys())
        self.key_pairs = [(task, opt) for task in self.task_keys for opt in self.opt_keys]
        self.active_pair = None
        self.num_samples = num_samples

        # initialize results dictionary
        self.results = {}
        print("Starting benchmarking!")
        self.cycle_task_optimizer_pair()

    def cycle_task_optimizer_pair(self) -> None:
        """Cycles through the next task/optimizer pair."""
        if self.key_pairs:
            print(f"    Pair: {self.key_pairs[0]}")

            # initialize empty results dict
            task_name, optimizer_name = self.key_pairs.pop(0)
            self.active_pair = (task_name, optimizer_name)
            self.results[(task_name, optimizer_name)] = {"plan_times": []}

            # instantiate the task and optimizer
            if task_name != self.active_pair[0]:
                self.node.send_output("task", pa.array([task_name]))
            if optimizer_name != self.active_pair[1]:
                self.node.send_output("optimizer", pa.array([optimizer_name]))
        else:
            self.print_results()
            sys.exit(0)

    def print_results(self) -> None:
        """Prints the results of the benchmarking using separate tables per task, with clean rules."""
        console = Console()
        print()

        # group results by task name
        grouped = defaultdict(list)
        for (task_name, optimizer_name), data in self.results.items():
            grouped[task_name].append((optimizer_name, data))

        # print results for each task
        for task_name, optimizers in grouped.items():
            table = Table(title=f"Results for Task: [bold]{task_name}[/bold]")
            table.add_column("Optimizer")
            table.add_column("Mean ± Std", justify="center")
            table.add_column("Median (IQR)", justify="center")
            table.add_column("Min / Max", justify="center")

            for optimizer_name, data in optimizers:
                # compute statistics for plan times
                plan_times = np.array(data["plan_times"])
                mean_time = np.mean(plan_times)
                std_time = np.std(plan_times)
                median_time = np.median(plan_times)
                iqr_25 = np.percentile(plan_times, 25)
                iqr_75 = np.percentile(plan_times, 75)
                min_time = np.min(plan_times)
                max_time = np.max(plan_times)

                table.add_row(
                    optimizer_name,
                    f"{mean_time:.4f} ± {std_time:.4f}",
                    f"{median_time:.4f} ({iqr_25:.4f}, {iqr_75:.4f})",
                    f"{min_time:.4f} / {max_time:.4f}",
                )

            console.print(table)
        print("Benchmarking complete! You may terminate the stack.")

    @on_event("INPUT", "plan_time")
    def on_plan_time(self, event: dict) -> None:
        """Handles the plan time event."""
        if self.active_pair is None and len(self.results[self.active_pair]["plan_times"]) < self.num_samples:
            return

        # append the plan time to the results
        plan_time = event["value"].to_numpy(zero_copy_only=False)[0]
        self.results[self.active_pair]["plan_times"].append(plan_time)

        # cycle to the next task/optimizer pair
        if len(self.results[self.active_pair]["plan_times"]) == self.num_samples:
            self.cycle_task_optimizer_pair()
