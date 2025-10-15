"""Main window of the Fleetmaster GUI."""

import sys

from PySide6.QtWidgets import QApplication, QLabel, QMainWindow


class MainWindow(QMainWindow):
    """Main window of the Fleetmaster GUI."""

    def __init__(self) -> None:
        """Initialize the main window."""
        super().__init__()
        self.setWindowTitle("Fleetmaster")
        self.setCentralWidget(QLabel("Hello, World!"))


def main() -> None:
    """Create and show the main window."""
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
