import sys
import os
from PyQt6.QtWidgets import QApplication
from ui_window import GalaxyClassifierApp
import app_config

if __name__ == "__main__":
    app_config.CURRENT_DIR = os.path.dirname(os.path.realpath(__file__))
    app = QApplication(sys.argv)
    window = GalaxyClassifierApp()
    window.show()
    sys.exit(app.exec())