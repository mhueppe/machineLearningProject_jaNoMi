# author: Michael Hüppe
# date: 28.10.2024
# project: /main_application.py

# local
import gui
import resources
# external
from PySide6.QtWidgets import QApplication, QMainWindow
from PySide6.QtGui import QIcon


class Headliner(QMainWindow):
    """
    Main Application class that uses Interface as the central widget.
    Includes internal methods for application logic.
    """

    def __init__(self):
        super().__init__()
        self.inputHandler = resources.model.JaNoMiModel()
        self.interface = gui.interface.Interface(
            cb_inputEnter=self._gui_inputerEnter,
            cb_getTokenizer=self.inputHandler.getTokenizer)  # Create an instance of Interface
        self.setCentralWidget(self.interface)  # Set Interface as the main widget
        self.initUI()
        self.setWindowTitle("Headliner")

    def initUI(self):
        """Initialize the main application UI settings."""
        self.setWindowTitle("Main Application")
        self.resize(800, 600)

    def _gui_inputerEnter(self, user_input: str, **kwargs) -> list:
        """
        Send the input of the gui to the input handler
        :return:
        """
        return self.inputHandler.generateOutput(user_input, **kwargs)


# Define the stylesheet
stylesheet = """
    QWidget {
        background-color: #D3D3D3;  /* Light grey background */
    }

    QPushButton {
        background-color: #cd2f2d;  /* Red background for buttons */
        color: black;               /* White text color */
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
    import sys
    app = QApplication(sys.argv)
    # app.setStyleSheet(stylesheet)
    # app.setStyleSheet("QWidget { color: black; background: none; }")
    # Set the application icon (modify the path to your icon file)
    icon_path = "gui/media/UHH_Universitaet_Hamburg_Logo.png"  # Provide the correct path to your PNG image
    # app.setStyle("Fusion")
    app.setWindowIcon(QIcon(icon_path))
    main_window = Headliner()
    main_window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
    # packaging command
    # pyinstaller -p "." --hidden-import gui --hidden-import resources --name Headliner --onefile --windowed --add-data "gui/media/UHH_Universitaet_Hamburg_Logo.png;gui/media" --add-data "vocabs;vocabs" --add-data "trained_models/Transformer/03_13_2025__15_19_56;trained_models/Transformer/03_13_2025__15_19_56" --add-data "trained_models/Transformer/03_13_2025__09_48_03;trained_models/Transformer/03_13_2025__09_48_03" --add-data "trained_models/TransformerDecoderOnly/03_14_2025__18_42_32;trained_models/TransformerDecoderOnly/03_14_2025__18_42_32" .\main.py
    # pyinstaller -p "." --hidden-import gui --hidden-import resources --name Headliner --add-data "gui/media/UHH_Universitaet_Hamburg_Logo.png;gui/media" --add-data "vocabs;vocabs" --add-data "trained_models/Transformer/03_13_2025__15_19_56;trained_models/Transformer/03_13_2025__15_19_56" --add-data "trained_models/Transformer/03_13_2025__09_48_03;trained_models/Transformer/03_13_2025__09_48_03" --add-data "trained_models/TransformerDecoderOnly/03_14_2025__18_42_32;trained_models/TransformerDecoderOnly/03_14_2025__18_42_32" .\main.py
