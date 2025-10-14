# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

import atexit
import os
import uuid
from pathlib import Path


def get_run_id(env_var: str = "JUDO_TASK_RUN_ID") -> str:
    """Return a per-process run-ID and whether it was freshly created."""
    run_id = os.environ.get(env_var)
    new_run = run_id is None
    if new_run:
        run_id = uuid.uuid4().hex
        os.environ[env_var] = run_id
    return run_id


def make_ephemeral_path(prefix: str, run_id: str, directory: str | Path = "/tmp") -> Path:
    """Create / clean a temp JSON path and register its teardown."""
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{prefix}_{run_id}.json"

    # remove stale siblings
    for old in directory.glob(f"{prefix}_*.json"):
        if old.name != path.name:
            try:
                old.unlink()
            except OSError:
                pass

    # ensure cleanup on interpreter exit
    def _cleanup() -> None:
        try:
            path.unlink()
        except FileNotFoundError:
            pass

    atexit.register(_cleanup)
    return path
