# fleetmaster/gui/handlers.py
import logging

from PyQt6.QtCore import QObject, pyqtSignal


class QtLogHandler(logging.Handler, QObject):
    """
    A custom logging handler that forwards log messages via a PyQt-signal.
    """

    message_written = pyqtSignal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__()
        QObject.__init__(self, parent)

    def emit(self, record: logging.LogRecord) -> None:
        """Invoked by the logger; transmits the signal."""
        msg = self.format(record)
        self.message_written.emit(msg)
