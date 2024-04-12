import sys
from PySide6.QtWidgets import QApplication, QMainWindow, QWidget, QGridLayout, QLabel

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Grid layout for central widget
        layout = QGridLayout(central_widget)

        # Placeholder widgets (you can replace these with any widgets you like)
        placeholder1 = QLabel("Placeholder 1")
        placeholder1.setStyleSheet("background-color: red;")

        placeholder2 = QLabel("Placeholder 2")
        placeholder2.setStyleSheet("background-color: green;")

        placeholder3 = QLabel("Placeholder 3")
        placeholder3.setStyleSheet("background-color: blue;")

        placeholder4 = QLabel("Placeholder 4")
        placeholder4.setStyleSheet("background-color: yellow;")

        # Adding widgets to layout with row, column, rowspan, and colspan
        layout.addWidget(placeholder1, 0, 0) # Top-left
        layout.addWidget(placeholder2, 0, 1) # Top-right
        layout.addWidget(placeholder3, 1, 0) # Bottom-left
        layout.addWidget(placeholder4, 1, 1) # Bottom-right

        # Adjusting the stretch factors to control the fractions
        # These factors do not correspond directly to percentage of area but affect the distribution of space
        layout.setColumnStretch(0, 3) # 30% width approximately
        layout.setColumnStretch(1, 2) # 20% width approximately
        layout.setRowStretch(0, 3) # 30% height approximately
        layout.setRowStretch(1, 2) # 20% height approximately

        self.setWindowTitle("Custom Layout")

if __name__ == "__main__":
    app = QApplication(sys.argv)

    window = MainWindow()
    window.resize(800, 600) # Set initial size
    window.show()

    sys.exit(app.exec())
