# fleetmaster/gui/handlers.py
import logging

from PySide6.QtCore import QObject, Signal


class _LogSignalEmitter(QObject):
    message_written = Signal(str)


class QtLogHandler(logging.Handler):
    """
    A custom logging handler that forwards log messages via a PyQt-signal.
    """

    def __init__(self) -> None:
        super().__init__()
        self.emitter = _LogSignalEmitter()
        self.message_written = self.emitter.message_written

    def emit(self, record: logging.LogRecord) -> None:
        """Invoked by the logger; transmits the signal."""
        msg = self.format(record)
        self.emitter.message_written.emit(msg)
