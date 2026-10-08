import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QFrame, QProgressBar, QSizePolicy, QMessageBox
)
from PyQt6.QtGui import QPixmap
from PyQt6.QtCore import Qt

from style import STYLE_SHEET
from image_loader import load_file
from predict import GalaxyInferenceEngine
import app_config


class GalaxyClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galaxy Spotter")
        self.resize(960, 700)
        self.setStyleSheet(STYLE_SHEET)

        self.current_tensor = None
        self.current_pixmap = None
        self.engine = GalaxyInferenceEngine()

        self._build_ui()
        self._check_engine_status()

    def _build_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(24, 20, 24, 24)
        main_layout.setSpacing(16)

        header_layout = QHBoxLayout()
        header_text = QVBoxLayout()
        title = QLabel("Galaxy Classifier")
        title.setObjectName("HeaderTitle")
        header_text.addWidget(title)

        self.status_badge = QLabel()
        self.status_badge.setStyleSheet("padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: bold;")

        header_layout.addLayout(header_text)
        header_layout.addStretch()
        header_layout.addWidget(self.status_badge)
        main_layout.addLayout(header_layout)

        content_layout = QHBoxLayout()
        content_layout.setSpacing(18)

        left_card = QFrame()
        left_card.setObjectName("Card")
        left_layout = QVBoxLayout(left_card)
        left_layout.setContentsMargins(18, 18, 18, 18)
        left_layout.setSpacing(14)

        image_title = QLabel("IMAGE VIEWER")
        image_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #64748b; letter-spacing: 1px;")
        left_layout.addWidget(image_title)

        self.preview_label = QLabel("No image loaded\nClick below to select a FITS or JPEG/PNG file")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #0b0f19; border: 2px dashed #233148; "
            "border-radius: 8px; color: #64748b; font-size: 13px;"
        )
        self.preview_label.setMinimumSize(380, 380)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.preview_label)

        btn_row = QHBoxLayout()
        self.upload_btn = QPushButton("Select Image (FITS / JPG)")
        self.upload_btn.clicked.connect(self._open_file_dialog)
        self.predict_btn = QPushButton("Classify Galaxy")
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self._run_prediction)

        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.predict_btn)
        left_layout.addLayout(btn_row)

        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #64748b; font-size: 11px;")
        left_layout.addWidget(self.file_info_label)

        content_layout.addWidget(left_card, stretch=5)

        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(16)

        results_header = QLabel("CLASSIFICATION REPORT")
        results_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #64748b; letter-spacing: 1px;")
        right_layout.addWidget(results_header)

        self.result_badge = QLabel("Awaiting Input")
        self.result_badge.setObjectName("ResultBadge")
        self.result_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_badge.setStyleSheet("background-color: #0f172a; color: #64748b;")
        right_layout.addWidget(self.result_badge)

        right_layout.addSpacing(10)

        conf_header = QLabel("Confidence Breakdown")
        conf_header.setStyleSheet("font-size: 13px; font-weight: bold; color: #cbd5e1;")
        right_layout.addWidget(conf_header)

        spiral_row = QHBoxLayout()
        spiral_title = QLabel("Spiral (S)")
        spiral_title.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.spiral_pct = QLabel("0.0%")
        self.spiral_pct.setStyleSheet("color: #38bdf8; font-weight: bold; font-size: 12px;")
        spiral_row.addWidget(spiral_title)
        spiral_row.addStretch()
        spiral_row.addWidget(self.spiral_pct)
        right_layout.addLayout(spiral_row)

        self.spiral_bar = QProgressBar()
        self.spiral_bar.setObjectName("SpiralBar")
        self.spiral_bar.setRange(0, 1000)
        self.spiral_bar.setValue(0)
        self.spiral_bar.setTextVisible(False)
        right_layout.addWidget(self.spiral_bar)

        right_layout.addSpacing(6)

        ellip_row = QHBoxLayout()
        ellip_title = QLabel("Elliptical (E)")
        ellip_title.setStyleSheet("color: #94a3b8; font-size: 12px;")
        self.elliptical_pct = QLabel("0.0%")
        self.elliptical_pct.setStyleSheet("color: #fbbf24; font-weight: bold; font-size: 12px;")
        ellip_row.addWidget(ellip_title)
        ellip_row.addStretch()
        ellip_row.addWidget(self.elliptical_pct)
        right_layout.addLayout(ellip_row)

        self.elliptical_bar = QProgressBar()
        self.elliptical_bar.setObjectName("EllipticalBar")
        self.elliptical_bar.setRange(0, 1000)
        self.elliptical_bar.setValue(0)
        self.elliptical_bar.setTextVisible(False)
        right_layout.addWidget(self.elliptical_bar)

        right_layout.addSpacing(10)

        meta_frame = QFrame()
        meta_frame.setStyleSheet("background-color: #0f172a; border-radius: 8px; padding: 12px;")
        meta_box = QVBoxLayout(meta_frame)
        meta_box.setSpacing(6)

        self.meta_details = QLabel("Image Specs: -")
        self.meta_device = QLabel(f"Device: {self.engine.device.type.upper()}")
        self.meta_weights = QLabel(f"Weights: {os.path.basename(app_config.WEIGHTS_PATH)}")

        for lbl in [self.meta_details, self.meta_device, self.meta_weights]:
            lbl.setStyleSheet("color: #94a3b8; font-size: 11px;")
            meta_box.addWidget(lbl)

        right_layout.addWidget(meta_frame)
        right_layout.addStretch()

        content_layout.addWidget(right_card, stretch=4)
        main_layout.addLayout(content_layout)

    def _check_engine_status(self):
        if self.engine.is_ready:
            self.status_badge.setText("WEIGHTS READY")
            self.status_badge.setStyleSheet("background-color: #064e3b; color: #34d399; padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: bold;")
        else:
            self.status_badge.setText("WEIGHTS NOT FOUND")
            self.status_badge.setStyleSheet("background-color: #7f1d1d; color: #f87171; padding: 6px 12px; border-radius: 6px; font-size: 11px; font-weight: bold;")
            self.result_badge.setText("Inference Disabled")
            self.result_badge.setStyleSheet("background-color: #450a0a; color: #fca5a5; font-size: 14px; padding: 10px; border-radius: 8px;")
            QMessageBox.warning(
                self,
                "Model Weights Missing",
                f"Could not load model weights.\n\nChecked path:\n{app_config.WEIGHTS_PATH}\n\n"
                "Please configure the valid full path in 'app_config.py' and restart the application."
            )

    def _open_file_dialog(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Galaxy Image",
            "",
            "All Files (*.fits *.fit *.fts *.jpg *.jpeg *.png);;FITS Files (*.fits *.fit *.fts);;Images (*.jpg *.jpeg *.png)"
        )
        if file_path:
            self._load_image(file_path)

    def _load_image(self, file_path):
        try:
            tensor, qimg, details = load_file(file_path)
            self.current_tensor = tensor
            self.current_pixmap = QPixmap.fromImage(qimg)
            self._render_pixmap()

            self.file_info_label.setText(f"File: {os.path.basename(file_path)}")
            self.meta_details.setText(details)

            if self.engine.is_ready:
                self.predict_btn.setEnabled(True)
                self.result_badge.setText("Ready to Classify")
                self.result_badge.setStyleSheet("background-color: #0f172a; color: #94a3b8;")
            else:
                self.predict_btn.setEnabled(False)

            self.spiral_bar.setValue(0)
            self.elliptical_bar.setValue(0)
            self.spiral_pct.setText("0.0%")
            self.elliptical_pct.setText("0.0%")

        except Exception as e:
            QMessageBox.critical(self, "Failed to Read Image", str(e))

    def _render_pixmap(self):
        if self.current_pixmap:
            scaled = self.current_pixmap.scaled(
                self.preview_label.size(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.preview_label.setPixmap(scaled)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._render_pixmap()

    def _run_prediction(self):
        if self.current_tensor is None or not self.engine.is_ready:
            return

        try:
            res = self.engine.predict(self.current_tensor)

            self.spiral_bar.setValue(int(res["prob_spiral"] * 1000))
            self.elliptical_bar.setValue(int(res["prob_elliptical"] * 1000))
            self.spiral_pct.setText(f"{res['prob_spiral'] * 100:.1f}%")
            self.elliptical_pct.setText(f"{res['prob_elliptical'] * 100:.1f}%")

            if res["pred_code"] == 1:
                self.result_badge.setText(f"SPIRAL GALAXY ({res['confidence'] * 100:.1f}%)")
                self.result_badge.setStyleSheet(
                    "background-color: #0369a1; color: #f0f9ff; border: 1px solid #38bdf8; "
                    "font-weight: bold; font-size: 16px; border-radius: 8px; padding: 10px;"
                )
            else:
                self.result_badge.setText(f"ELLIPTICAL GALAXY ({res['confidence'] * 100:.1f}%)")
                self.result_badge.setStyleSheet(
                    "background-color: #b45309; color: #fffbeb; border: 1px solid #fbbf24; "
                    "font-weight: bold; font-size: 16px; border-radius: 8px; padding: 10px;"
                )
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))
