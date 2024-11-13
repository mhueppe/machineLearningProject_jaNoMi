# author: Michael Hüppe
# date: 28.10.2024
# project: /main_application.py
from PySide2.QtGui import QIcon

# local
from gui.interface import Interface
from resources.model import JaNoMiModel
# external
from PySide2.QtWidgets import QApplication, QMainWindow
import sys
from resources.model_types import ModelTypes

class MainApplication(QMainWindow):
    """
    Main Application class that uses Interface as the central widget.
    Includes internal methods for application logic.
    """

    def __init__(self):
        super().__init__()
        self.interface = Interface(cb_inputEnter=self._gui_inputerEnter)  # Create an instance of Interface
        self.inputHandler = JaNoMiModel()
        self.setCentralWidget(self.interface)  # Set Interface as the main widget
        self.initUI()
        self.setWindowTitle("Headliner")

    def initUI(self):
        """Initialize the main application UI settings."""
        self.setWindowTitle("Main Application")
        self.resize(800, 600)

    def _gui_inputerEnter(self, userInput: str, modelType: ModelTypes) -> None:
        """
        Send the input of the gui to the input handler
        :return:
        """
        output = self.inputHandler.generateOutput(userInput, modelType)
        self.interface.handleOutput(output)


# Define the stylesheet
stylesheet = """
    QWidget {
        background-color: #D3D3D3;  /* Light grey background */
    }

    QPushButton {
        background-color: #cd2f2d;  /* Red background for buttons */
        color: white;               /* White text color */
        border-radius: 15px;        /* Rounded edges */
        padding: 10px;              /* Internal padding */
        font-size: 16px;            /* Font size */
    }

    QPushButton:hover {
        background-color: #CC0000; /* Darker red on hover */
    }

    QPushButton:pressed {
        background-color: #990000; /* Even darker red on press */
    }
"""

def main():
    """Main entry point for the application."""
    app = QApplication(sys.argv)
    app.setStyleSheet(stylesheet)
    # Set the application icon (modify the path to your icon file)
    icon_path = "gui/media/UHH_Universitaet_Hamburg_Logo.png"  # Provide the correct path to your PNG image
    app.setStyle("Fusion")
    app.setWindowIcon(QIcon(icon_path))
    main_window = MainApplication()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
