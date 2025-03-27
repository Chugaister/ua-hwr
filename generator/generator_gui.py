import sys
from PyQt6.QtWidgets import QApplication, QWidget, QVBoxLayout, QLineEdit, QLabel
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt
from PIL.ImageQt import ImageQt
from generator import HandwritingGenerator


class TextToImageApp(QWidget):
    def __init__(self):
        super().__init__()

        # Initialize the handwriting generator with dataset
        self.generator = HandwritingGenerator("glyphs.csv")

        self.setWindowTitle("Text to Image Generator")
        self.setGeometry(100, 100, 500, 600)

        # Layout
        self.layout = QVBoxLayout()

        # Text input
        self.text_input = QLineEdit(self)
        self.text_input.setPlaceholderText("Enter text here...")
        self.text_input.textChanged.connect(self.generate_image)  # Call on text change
        self.layout.addWidget(self.text_input)

        # Image display label
        self.image_label = QLabel(self)
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.layout.addWidget(self.image_label)

        self.setLayout(self.layout)

    def generate_image(self):
        text = self.text_input.text().strip()
        if text:
            pil_image = self.generator.generate(text)  # Generate the image
            qt_image = ImageQt(pil_image)  # Convert PIL Image to QImage
            pixmap = QPixmap.fromImage(qt_image)  # Convert to QPixmap
            self.image_label.setPixmap(pixmap)  # Display image
            self.image_label.adjustSize()  # Ensure it doesn't resize automatically


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = TextToImageApp()
    window.show()
    sys.exit(app.exec())
