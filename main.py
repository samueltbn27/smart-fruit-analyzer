from PyQt5.QtWidgets import (QApplication, QMainWindow, QLabel, QVBoxLayout, QHBoxLayout,
                             QWidget, QPushButton, QFileDialog, QGroupBox, QComboBox)
from PyQt5.QtGui import QImage, QPixmap, QFont
from PyQt5.QtCore import Qt, QSize # Import QSize for explicit sizing
import cv2
import numpy as np
import sys
import os

# Import functions from other files
from image_processor import preprocess_image, detect_edges
from fruit_detector import color_based_detection, detect_banana_ripeness

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SmartFruit Analyzer Pro")
        # Adjust window size to accommodate two images side-by-side
        self.setFixedSize(1600, 850) # Increased width and slight height increase
        self.setup_ui()
        self.original_cv_img = None # Stores the original loaded OpenCV image
        self.processed_cv_img = None # Stores the processed OpenCV image

    def setup_ui(self):
        # Image Display Labels (Original and Processed)
        self.original_image_display_label = QLabel("Original Image Area")
        self.original_image_display_label.setAlignment(Qt.AlignCenter)
        self.original_image_display_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.original_image_display_label.setMinimumSize(750, 550) # Min size for original image
        # self.original_image_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.processed_image_display_label = QLabel("Processed Image Area")
        self.processed_image_display_label.setAlignment(Qt.AlignCenter)
        self.processed_image_display_label.setStyleSheet("border: 2px solid #ccc; background-color: #f0f0f0;")
        self.processed_image_display_label.setMinimumSize(750, 550) # Min size for processed image
        # self.processed_image_display_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.processed_image_info_label = QLabel("No processing applied yet.")
        self.processed_image_info_label.setAlignment(Qt.AlignCenter)
        font_info = QFont("Arial", 9)
        self.processed_image_info_label.setFont(font_info)
        self.processed_image_info_label.setFixedHeight(30) # Give it a fixed height

        # Layout for the processed image area (image + info label)
        processed_area_layout = QVBoxLayout()
        processed_area_layout.addWidget(self.processed_image_display_label, 1) # Image display gets more stretch
        processed_area_layout.addWidget(self.processed_image_info_label, 0) # Info label takes less space

        # Layout for Image Displays
        image_display_layout = QHBoxLayout()
        image_display_layout.addWidget(self.original_image_display_label, 1) # Stretch factor 1
        image_display_layout.addLayout(processed_area_layout, 1) # Add the layout for processed area

        # RTM_2 Operations
        self.gbox_rtm2 = QGroupBox("Image Processing (RTM_2)")
        gbox_font = QFont("Arial", 10, QFont.Bold)
        self.gbox_rtm2.setFont(gbox_font)
        btn_load = QPushButton("📁 Load Image")
        btn_grayscale = QPushButton("Grayscale")
        btn_contrast = QPushButton("Enhance Contrast")
        btn_edges = QPushButton("Detect Edges")

        # RTM3 Detection
        self.gbox_rtm3 = QGroupBox("Fruit Detection (RTM3)")
        self.gbox_rtm3.setFont(gbox_font)
        self.cb_fruit_type = QComboBox()
        self.cb_fruit_type.addItems(["All Fruits", "Apple", "Orange", "Banana"])
        btn_detect = QPushButton("Detect Fruits")
        btn_ripeness = QPushButton("Check Banana Ripeness")

        # Layouts
        layout_rtm2 = QHBoxLayout()
        layout_rtm2.addWidget(btn_load)
        layout_rtm2.addWidget(btn_grayscale)
        layout_rtm2.addWidget(btn_contrast)
        layout_rtm2.addWidget(btn_edges)

        layout_rtm3 = QHBoxLayout()
        layout_rtm3.addWidget(self.cb_fruit_type)
        layout_rtm3.addWidget(btn_detect)
        layout_rtm3.addWidget(btn_ripeness)

        self.gbox_rtm2.setLayout(layout_rtm2)
        self.gbox_rtm3.setLayout(layout_rtm3)

        # Main Layout
        main_layout = QVBoxLayout()
        main_layout.addLayout(image_display_layout, 1) # Add the QHBoxLayout for images, give it stretch
        main_layout.addWidget(self.gbox_rtm2)
        main_layout.addWidget(self.gbox_rtm3)

        container = QWidget()
        container.setLayout(main_layout)
        self.setCentralWidget(container)

        # Signal Connections
        btn_load.clicked.connect(self.load_image)
        btn_grayscale.clicked.connect(self.apply_grayscale)
        btn_contrast.clicked.connect(self.enhance_contrast)
        btn_edges.clicked.connect(self.apply_edge_detection)
        btn_detect.clicked.connect(self.detect_fruits)
        btn_ripeness.clicked.connect(self.check_banana_ripeness)

    def load_image(self):
        file_path, _ = QFileDialog.getOpenFileName(filter="Images (*.jpg *.png *.jpeg)") # Added .jpeg
        if file_path:
            self.original_cv_img = cv2.imread(file_path)
            if self.original_cv_img is not None:
                self.display_cv_image(self.original_cv_img, self.original_image_display_label)
                # Clear the processed image display and reset processed image
                self.processed_image_display_label.clear()
                self.processed_image_display_label.setText("Processed image will appear here.")
                self.processed_image_info_label.setText("Image loaded. Select a process.")
                self.processed_cv_img = None
            else:
                print(f"Error: Could not load image from {file_path}")
                self.original_image_display_label.setText("Failed to load image.")
                self.processed_image_display_label.clear()
                self.processed_image_info_label.setText("Failed to load image.")

    def apply_grayscale(self):
        if self.original_cv_img is not None:
            gray = cv2.cvtColor(self.original_cv_img, cv2.COLOR_BGR2GRAY)
            self.processed_cv_img = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR) # Convert back for display
            self.display_cv_image(self.processed_cv_img, self.processed_image_display_label)
            self.processed_image_info_label.setText("Applied: Grayscale Conversion")
        else:
            self.processed_image_display_label.setText("Load an image first to apply grayscale.")
            self.processed_image_info_label.setText("Load an image first.")

    def enhance_contrast(self):
        if self.original_cv_img is not None:
            enhanced = preprocess_image(self.original_cv_img.copy()) # Pass a copy if preprocess modifies
            self.processed_cv_img = enhanced
            self.display_cv_image(self.processed_cv_img, self.processed_image_display_label)
            self.processed_image_info_label.setText("Applied: Contrast Enhancement (CLAHE)")
        else:
            self.processed_image_display_label.setText("Load an image first to enhance contrast.")
            self.processed_image_info_label.setText("Load an image first.")

    def apply_edge_detection(self):
        if self.original_cv_img is not None:
            edges = detect_edges(self.original_cv_img.copy()) # Pass a copy
            self.processed_cv_img = edges
            self.display_cv_image(self.processed_cv_img, self.processed_image_display_label)
            self.processed_image_info_label.setText("Applied: Edge Detection (Canny)")
        else:
            self.processed_image_display_label.setText("Load an image first to detect edges.")
            self.processed_image_info_label.setText("Load an image first.")

    def detect_fruits(self):
        if self.original_cv_img is not None:
            fruit_type = self.cb_fruit_type.currentText()
            # Pass a copy of the original image for detection
            result = color_based_detection(self.original_cv_img.copy(), fruit_type)
            self.processed_cv_img = result
            self.display_cv_image(self.processed_cv_img, self.processed_image_display_label)
            self.processed_image_info_label.setText(f"Applied: Fruit Detection ({fruit_type})")
        else:
            self.processed_image_display_label.setText("Load an image first to detect fruits.")
            self.processed_image_info_label.setText("Load an image first.")

    def check_banana_ripeness(self):
        if self.original_cv_img is not None:
            # Pass a copy of the original image for ripeness detection
            result = detect_banana_ripeness(self.original_cv_img.copy())
            self.processed_cv_img = result
            self.display_cv_image(self.processed_cv_img, self.processed_image_display_label)
            self.processed_image_info_label.setText("Applied: Banana Ripeness Check")
        else:
            self.processed_image_display_label.setText("Load an image first to check banana ripeness.")
            self.processed_image_info_label.setText("Load an image first.")

    def display_cv_image(self, cv_img, target_label_widget):
        if cv_img is None:
            target_label_widget.clear()
            target_label_widget.setText("No image to display.")
            return

        # Convert to RGB for QImage
        # Handle potential single channel images (like Canny edges output)
        if len(cv_img.shape) == 2 or cv_img.shape[2] == 1: # Grayscale or single channel
            cv_img_display = cv2.cvtColor(cv_img, cv2.COLOR_GRAY2RGB) if len(cv_img.shape) == 2 else cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB) # Canny might return BGR
        else:
            cv_img_display = cv2.cvtColor(cv_img, cv2.COLOR_BGR2RGB)

        img_rgb = cv_img_display # Use the prepared image for QImage
        h, w, ch = img_rgb.shape
        bytes_per_line = ch * w
        q_img = QImage(img_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        # Scale the QPixmap to fit the QLabel while maintaining aspect ratio
        # Use image_label.size() for more responsive scaling
        pixmap = QPixmap.fromImage(q_img)
        target_label_widget.setPixmap(pixmap.scaled(
            target_label_widget.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))
        target_label_widget.setAlignment(Qt.AlignCenter) # Ensure alignment is set after pixmap

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())