import sys
import pandas as pd
from PyQt6.QtWidgets import QApplication, QLabel, QPushButton, QFileDialog, QVBoxLayout, QWidget
from PyQt6.QtGui import QPixmap, QImage, QPainter, QPen, QCursor
from PyQt6.QtCore import Qt, QRect
import cv2


class HandwritingAnnotator(QWidget):
    def __init__(self):
        super().__init__()
        self.dataset = None
        self.current_index = 0
        self.rect = QRect()
        self.start_pos = None
        self.end_pos = None
        self.image_label = QLabel(self)
        self.info_label = QLabel(self)
        self.next_button = QPushButton("Next", self)
        self.save_button = QPushButton("Save", self)
        self.uppercase_button = QPushButton("Uppercase", self)

        self.next_button.clicked.connect(self.load_next_image)
        self.save_button.clicked.connect(self.save_rectangle)
        self.uppercase_button.clicked.connect(self.apply_uppercase)

        layout = QVBoxLayout()
        layout.addWidget(self.image_label)
        layout.addWidget(self.info_label)
        layout.addWidget(self.next_button)
        layout.addWidget(self.save_button)
        self.setLayout(layout)

        self.setCursor(QCursor(Qt.CursorShape.CrossCursor))

        self.load_csv()
        self.load_next_image()

    def load_csv(self):
        file_path, _ = QFileDialog.getOpenFileName(self, "Open CSV File", "", "CSV Files (*.csv)")
        if file_path:
            self.dataset = pd.read_csv(file_path)

    def apply_uppercase(self):
        self.dataset.loc[self.dataset["is_uppercase"], "label"] = self.dataset.loc[self.dataset["is_uppercase"], "label"].str.upper()

    def load_next_image(self):
        if self.dataset is None:
            return

        while self.current_index < len(self.dataset):
            row = self.dataset.iloc[self.current_index]
            if not pd.isna(row['btop']):  # Skip already annotated letters
                self.current_index += 1
                continue

            img_path = row['filename']
            self.current_label = row['label']
            self.is_uppercase = row['is_uppercase']
            self.current_image = cv2.imread(img_path, cv2.IMREAD_UNCHANGED)
            if self.current_image is None:
                self.current_index += 1
                continue

            self.display_image()
            self.info_label.setText(f"Character: {self.current_label} (Uppercase: {self.is_uppercase})")
            self.current_index += 1
            return

    def display_image(self):
        height, width, channel = self.current_image.shape
        bytes_per_line = channel * width
        q_img = QImage(self.current_image.data, width, height, bytes_per_line, QImage.Format.Format_BGR888)
        pixmap = QPixmap.fromImage(q_img)

        painter = QPainter(pixmap)
        painter.setPen(QPen(Qt.GlobalColor.red, 2, Qt.PenStyle.SolidLine))
        if not self.rect.isNull():
            painter.drawRect(self.rect)

        # Draw dots on top, bottom, left, right coordinates if available
        row = self.dataset.iloc[self.current_index - 1]
        if not pd.isna(row['top']):
            painter.setPen(QPen(Qt.GlobalColor.blue, 5, Qt.PenStyle.SolidLine))
            painter.drawPoint(int(row['top']), int(row['left']))
            painter.drawPoint(int(row['top']), int(row['right']))
            painter.drawPoint(int(row['bottom']), int(row['left']))
            painter.drawPoint(int(row['bottom']), int(row['right']))

        painter.end()
        self.image_label.setPixmap(pixmap)

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.start_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
            self.rect = QRect()

    def mouseMoveEvent(self, event):
        if self.start_pos is not None:
            self.end_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
            self.rect = QRect(self.start_pos, self.end_pos)
            self.display_image()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.rect is not None:
            self.end_pos = self.image_label.mapFromGlobal(event.globalPosition().toPoint())
            self.display_image()

    def save_rectangle(self):
        if self.dataset is None or self.current_index == 0:
            return
        row_idx = self.current_index - 1
        self.dataset.at[row_idx, 'btop'] = self.rect.top()
        self.dataset.at[row_idx, 'bbottom'] = self.rect.bottom()
        self.dataset.at[row_idx, 'bleft'] = self.rect.left()
        self.dataset.at[row_idx, 'bright'] = self.rect.right()
        self.dataset.to_csv('updated_dataset.csv', index=False)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = HandwritingAnnotator()
    window.show()
    sys.exit(app.exec())
