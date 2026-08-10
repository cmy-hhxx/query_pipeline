"""Per-command structured logging with stable batch identities."""

from __future__ import annotations

import contextvars
import logging
import re
import secrets
import sys
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from types import TracebackType
from typing import Literal

from filelock import FileLock, Timeout

from query_pipeline.io.jsonl import dumps_jsonl

_BEIJING = timezone(timedelta(hours=8))
_SAFE_BATCH_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._+\-]{0,127}$")
_ACTIVE_CONTEXT: contextvars.ContextVar[tuple[str, str] | None] = contextvars.ContextVar(
    "query_pipeline_log_context", default=None
)
_LOGGER_STATE_LOCK = threading.Lock()
_LOGGER_SESSIONS = 0
_LOGGER_ORIGINAL_STATE: tuple[int, bool] | None = None


def generate_batch_id() -> str:
    timestamp = datetime.now(tz=_BEIJING).strftime("%Y%m%dT%H%M%S%z")
    return f"{timestamp}_{secrets.token_hex(4)}"


def validate_batch_id(value: str | None) -> str:
    batch_id = generate_batch_id() if value is None else value
    if batch_id in {".", ".."} or not _SAFE_BATCH_ID.fullmatch(batch_id):
        raise ValueError(
            "batch_id must be 1-128 filename-safe characters: letters, digits, '.', '_', '+', '-'"
        )
    return batch_id


def beijing_timestamp(timestamp: float | None = None) -> str:
    moment = (
        datetime.now(tz=_BEIJING)
        if timestamp is None
        else datetime.fromtimestamp(timestamp, tz=_BEIJING)
    )
    return moment.isoformat(timespec="milliseconds")


class _BatchFilter(logging.Filter):
    def __init__(self, command: str, batch_id: str) -> None:
        super().__init__()
        self.command = command
        self.batch_id = batch_id

    def filter(self, record: logging.LogRecord) -> bool:
        if _ACTIVE_CONTEXT.get() != (self.command, self.batch_id):
            return False
        record.command = self.command
        record.batch_id = self.batch_id
        return True


class JsonLogFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        payload = {
            "timestamp": beijing_timestamp(record.created),
            "level": record.levelname,
            "logger": record.name,
            "command": getattr(record, "command", ""),
            "batch_id": getattr(record, "batch_id", ""),
            "message": record.getMessage(),
        }
        if record.exc_info:
            payload["exception"] = self.formatException(record.exc_info)
        return dumps_jsonl(payload)


class BeijingTextFormatter(logging.Formatter):
    def formatTime(  # noqa: N802
        self, record: logging.LogRecord, datefmt: str | None = None
    ) -> str:
        moment = datetime.fromtimestamp(record.created, tz=_BEIJING)
        if datefmt:
            return moment.strftime(datefmt)
        return moment.isoformat(timespec="seconds")


class _FailFastFileHandler(logging.FileHandler):
    def handleError(self, record: logging.LogRecord) -> None:  # noqa: N802
        error = sys.exc_info()[1]
        if error is not None:
            raise error
        raise OSError(f"failed to write ordinary log record: {record.getMessage()}")


class LoggingSession:
    """Attach isolated file/console handlers for one command batch."""

    def __init__(
        self,
        log_dir: str | Path,
        *,
        command: str,
        batch_id: str | None = None,
        verbose: bool = False,
    ) -> None:
        self.log_dir = Path(log_dir).expanduser().resolve()
        self.command = command
        self.batch_id = validate_batch_id(batch_id)
        self.ordinary_path = self.log_dir / "ordinary" / command / f"{self.batch_id}.log"
        self.verbose = verbose
        self.logger = logging.getLogger("query_pipeline")
        self._batch_lock: FileLock | None = None
        self._file_handler: logging.Handler | None = None
        self._console_handler: logging.Handler | None = None
        self._context_token: contextvars.Token[tuple[str, str] | None] | None = None
        self._attached = False

    def __enter__(self) -> LoggingSession:
        self.ordinary_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path = self.log_dir / ".locks" / self.command / f"{self.batch_id}.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        self._batch_lock = FileLock(lock_path, timeout=0)
        try:
            self._batch_lock.acquire()
        except Timeout as exc:
            self._batch_lock = None
            raise RuntimeError(
                f"command batch is already running: command={self.command} batch_id={self.batch_id}"
            ) from exc

        try:
            batch_filter = _BatchFilter(self.command, self.batch_id)
            level = logging.DEBUG if self.verbose else logging.INFO
            file_handler = _FailFastFileHandler(self.ordinary_path, encoding="utf-8")
            file_handler.setLevel(level)
            file_handler.addFilter(batch_filter)
            file_handler.setFormatter(JsonLogFormatter())
            console_handler = logging.StreamHandler()
            console_handler.setLevel(level)
            console_handler.addFilter(batch_filter)
            console_handler.setFormatter(
                BeijingTextFormatter("%(asctime)s %(levelname)s %(name)s: %(message)s")
            )
            self._file_handler = file_handler
            self._console_handler = console_handler

            global _LOGGER_SESSIONS, _LOGGER_ORIGINAL_STATE
            with _LOGGER_STATE_LOCK:
                if _LOGGER_SESSIONS == 0:
                    _LOGGER_ORIGINAL_STATE = (self.logger.level, self.logger.propagate)
                    self.logger.setLevel(logging.DEBUG)
                    self.logger.propagate = False
                _LOGGER_SESSIONS += 1
                self.logger.addHandler(file_handler)
                self.logger.addHandler(console_handler)
                self._attached = True

            self._context_token = _ACTIVE_CONTEXT.set((self.command, self.batch_id))
            self.logger.info("command_started ordinary_log=%s", self.ordinary_path)
        except Exception:
            if self._context_token is not None:
                _ACTIVE_CONTEXT.reset(self._context_token)
                self._context_token = None
            try:
                self._detach_handlers()
            finally:
                self._release_batch_lock()
            raise
        return self

    def __exit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        try:
            if exc is None:
                self.logger.info("command_finished")
            else:
                self.logger.error(
                    "command_failed: %s", exc, exc_info=(type(exc), exc, traceback)
                )
        finally:
            if self._context_token is not None:
                _ACTIVE_CONTEXT.reset(self._context_token)
                self._context_token = None
            try:
                self._detach_handlers()
            finally:
                self._release_batch_lock()
        return False

    def _release_batch_lock(self) -> None:
        if self._batch_lock is not None:
            self._batch_lock.release()
            self._batch_lock = None

    def _detach_handlers(self) -> None:
        global _LOGGER_SESSIONS, _LOGGER_ORIGINAL_STATE
        close_error: Exception | None = None
        with _LOGGER_STATE_LOCK:
            for handler in (self._file_handler, self._console_handler):
                if handler is not None:
                    self.logger.removeHandler(handler)
                    try:
                        handler.close()
                    except Exception as exc:
                        close_error = close_error or exc
            self._file_handler = None
            self._console_handler = None
            if not self._attached:
                if close_error is not None:
                    raise close_error
                return
            try:
                self._attached = False
                _LOGGER_SESSIONS -= 1
                if _LOGGER_SESSIONS == 0 and _LOGGER_ORIGINAL_STATE is not None:
                    level, propagate = _LOGGER_ORIGINAL_STATE
                    self.logger.setLevel(level)
                    self.logger.propagate = propagate
                    _LOGGER_ORIGINAL_STATE = None
            finally:
                if close_error is not None:
                    raise close_error


def logging_session(
    log_dir: str | Path,
    *,
    command: str,
    batch_id: str | None = None,
    verbose: bool = False,
) -> LoggingSession:
    return LoggingSession(log_dir, command=command, batch_id=batch_id, verbose=verbose)
