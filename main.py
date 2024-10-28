# author: Michael Hüppe
# date: 28.10.2024
# project: /main_application.py

# local
from gui.interface import Interface
from resources.inputHandler import InputHandler
# external
from PySide2.QtWidgets import QApplication, QMainWindow
import sys


class MainApplication(QMainWindow):
    """
    Main Application class that uses Interface as the central widget.
    Includes internal methods for application logic.
    """

    def __init__(self):
        super().__init__()
        self.interface = Interface(cb_inputEnter=self._gui_inputerEnter)  # Create an instance of Interface
        self.inputHandler = InputHandler()
        self.setCentralWidget(self.interface)  # Set Interface as the main widget
        self.initUI()
        self.setWindowTitle("JaNoMi Machine Learning Project")

    def initUI(self):
        """Initialize the main application UI settings."""
        self.setWindowTitle("Main Application")
        self.resize(800, 600)

    def _gui_inputerEnter(self, userInput: str) -> None:
        """
        Send the input of the gui to the input handler
        :return:
        """
        output = self.inputHandler.handleInput(userInput)
        self.interface.handleOutput(output)


def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    main_window = MainApplication()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
