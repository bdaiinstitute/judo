# Copyright (c) 2025 Robotics and AI Institute LLC. All rights reserved.

# written by Duy
import logging
import sys

import colorlog

DEFAULT_DATE_FORMAT = "%H:%M:%S"
DEFAULT_FORMAT = "%(log_color)s[%(levelname)s][%(asctime)s][%(processName)s][%(module)s]: %(message)s"
DEFAULT_MCAP_FORMAT = "[%(module)s] %(message)s"
DEFAULT_THREAD_FORMAT = (
    "%(log_color)s[%(levelname)s][%(asctime)s][%(processName)s][%(threadName)s][%(module)s]: %(message)s"
)

LOG_COLORS = {
    "DEBUG": "light_black",
    "INFO": "white",
    "WARNING": "yellow",
    "ERROR": "red",
    "CRITICAL": "red",
}


def get_format(include_thread: bool) -> str:
    return DEFAULT_THREAD_FORMAT if include_thread else DEFAULT_FORMAT


def setup_logger(log_level: str | int = logging.INFO, include_thread: bool = False) -> None:
    """Sets root logger. You should only run this once per node/runtime."""
    colorlog.basicConfig(
        level=log_level,
        format=get_format(include_thread),
        datefmt=DEFAULT_DATE_FORMAT,
        # Force is required to make sure that any other instantiations/setting
        # of a logger don't prevent this from going into effect. Even just calling
        # logging.getLogger() will prevent subsequent logging.basicConfig() calls to be
        # ignored unless the "force" flag it True.
        force=True,
    )

    # Set custom colors so all the INFO messages aren't obnoxiously green.
    # For some reason the custom colors don't set properly when done as part of the basicConfig call
    formatter = colorlog.ColoredFormatter(
        get_format(include_thread),
        log_colors=LOG_COLORS,
        reset=True,
    )
    handler = colorlog.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)
    logger = colorlog.getLogger()
    logger.handlers.clear()
    logger.addHandler(handler)


# We may want this to take in a custom format string eventually
def configure_local_logger_format(logger: logging.Logger, include_thread: bool = False) -> None:
    """Configures a specific logger instance to be different from the root logger."""
    root_logger = colorlog.getLogger()
    # Because the root logger may be using force=True, you need to clear associatings of this
    # logger from the root logger
    logger.handlers.clear()
    handler = colorlog.StreamHandler(sys.stdout)
    formatter = colorlog.ColoredFormatter(
        fmt=get_format(include_thread),
        log_colors=LOG_COLORS,
        datefmt=DEFAULT_DATE_FORMAT,
    )
    handler.setFormatter(formatter)
    logger.addHandler(handler)
    logger.propagate = False
    logger.setLevel(root_logger.getEffectiveLevel())
