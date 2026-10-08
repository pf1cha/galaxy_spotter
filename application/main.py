import sys
from PyQt6.QtWidgets import QApplication
from ui_window import GalaxyClassifierApp


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = GalaxyClassifierApp()
    window.show()
    sys.exit(app.exec())