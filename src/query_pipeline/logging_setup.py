"""Central logging setup: Beijing-time timestamps everywhere, log to file+stream.

`setup_logging` attaches a stream handler and a file handler (swapped per run
to the current output directory) and forces every logging Formatter in the
process to render timestamps in Asia/Shanghai — including third-party loggers
(openai/httpx/...) so the whole log surface is Beijing time.
"""

from __future__ import annotations

import logging
import time
from datetime import datetime, timedelta, timezone
from pathlib import Path

_BEIJING = timezone(timedelta(hours=8))


def beijing_converter(timestamp: float) -> time.struct_time:
    return datetime.fromtimestamp(timestamp, tz=_BEIJING).timetuple()


def setup_logging(log_file: Path, *, verbose: bool) -> logging.Logger:
    """Configure the query_pipeline logger: stream + per-run file handler."""
    # Beijing time for every formatter in the process (own + imported loggers).
    setattr(logging.Formatter, "converter", staticmethod(beijing_converter))

    logger = logging.getLogger("query_pipeline")
    logger.setLevel(logging.DEBUG if verbose else logging.INFO)
    logger.propagate = False
    for handler in list(logger.handlers):
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
    if not any(isinstance(h, logging.StreamHandler) for h in logger.handlers):
        stream = logging.StreamHandler()
        stream.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        logger.addHandler(stream)
    log_file.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_file, encoding="utf-8")
    file_handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    logger.addHandler(file_handler)
    return logger


def beijing_now() -> str:
    return datetime.now(tz=_BEIJING).strftime("%Y-%m-%d %H:%M:%S")
