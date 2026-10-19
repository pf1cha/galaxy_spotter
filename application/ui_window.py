import os
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QLabel, QPushButton, QVBoxLayout,
    QHBoxLayout, QFileDialog, QFrame, QProgressBar, QSizePolicy, QMessageBox,
    QDialog, QFormLayout, QLineEdit, QComboBox, QDialogButtonBox
)
from PyQt6.QtGui import QPixmap, QIcon
from PyQt6.QtCore import Qt

from style import STYLE_SHEET
from image_loader import load_file, get_fits_cutout
from predict import GalaxyInferenceEngine
import app_config


class GalaxyClassifierApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Galaxy Spotter")
        self.setWindowIcon(QIcon(os.path.dirname(os.path.realpath(__file__)) + os.path.sep + 'logo.png'))
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
        subtitle = QLabel("Explore galaxy classification with your own object")
        subtitle.setObjectName("HeaderSubtitle")
        header_text.addWidget(subtitle)

        self.status_badge = QLabel()
        self.status_badge.setStyleSheet(
            "background-color: #302039; color: #d9a9e8; padding: 6px 12px; "
            "border: 1px solid #563763; border-radius: 6px; font-size: 11px; font-weight: bold;"
        )
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
        image_title.setStyleSheet("font-size: 12px; font-weight: bold; color: #c08bd0; letter-spacing: 1px;")
        left_layout.addWidget(image_title)

        self.preview_label = QLabel("No image loaded\nClick below to select a FITS image")
        self.preview_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.preview_label.setStyleSheet(
            "background-color: #110d19; border: 2px dashed #563763; "
            "border-radius: 8px; color: #a995b4; font-size: 13px;"
        )
        self.preview_label.setMinimumSize(380, 380)
        self.preview_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        left_layout.addWidget(self.preview_label)

        btn_row = QHBoxLayout()
        self.upload_btn = QPushButton("Select FITS Image")
        self.upload_btn.clicked.connect(self._open_file_dialog)
        self.coords_btn = QPushButton("Request by Coordinates")
        self.coords_btn.clicked.connect(self._open_coords_dialog)
        self.predict_btn = QPushButton("Classify Galaxy")
        self.predict_btn.setEnabled(False)
        self.predict_btn.clicked.connect(self._run_prediction)

        btn_row.addWidget(self.upload_btn)
        btn_row.addWidget(self.coords_btn)
        btn_row.addWidget(self.predict_btn)
        left_layout.addLayout(btn_row)

        self.file_info_label = QLabel("No file selected")
        self.file_info_label.setStyleSheet("color: #a995b4; font-size: 11px;")
        left_layout.addWidget(self.file_info_label)

        content_layout.addWidget(left_card, stretch=5)

        right_card = QFrame()
        right_card.setObjectName("Card")
        right_layout = QVBoxLayout(right_card)
        right_layout.setContentsMargins(20, 20, 20, 20)
        right_layout.setSpacing(16)

        results_header = QLabel("CLASSIFICATION REPORT")
        results_header.setStyleSheet("font-size: 12px; font-weight: bold; color: #c08bd0; letter-spacing: 1px;")
        right_layout.addWidget(results_header)

        self.result_badge = QLabel("Awaiting Input")
        self.result_badge.setObjectName("ResultBadge")
        self.result_badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.result_badge.setStyleSheet(
            "background-color: #21182a; color: #b9a7c3; border: 1px solid #392a43;"
        )
        right_layout.addWidget(self.result_badge)

        right_layout.addSpacing(10)

        conf_header = QLabel("Confidence Breakdown")
        conf_header.setStyleSheet("font-size: 13px; font-weight: bold; color: #f0e4f4;")
        right_layout.addWidget(conf_header)

        spiral_row = QHBoxLayout()
        spiral_title = QLabel("Spiral (S)")
        spiral_title.setStyleSheet("color: #c7b4d0; font-size: 12px;")
        self.spiral_pct = QLabel("0.0%")
        self.spiral_pct.setStyleSheet("color: #B120D9; font-weight: bold; font-size: 12px;")
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
        ellip_title.setStyleSheet("color: #c7b4d0; font-size: 12px;")
        self.elliptical_pct = QLabel("0.0%")
        self.elliptical_pct.setStyleSheet("color: #F37A15; font-weight: bold; font-size: 12px;")
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
        meta_frame.setStyleSheet(
            "background-color: #21182a; border: 1px solid #392a43; border-radius: 8px; padding: 12px;"
        )
        meta_box = QVBoxLayout(meta_frame)
        meta_box.setSpacing(6)

        self.meta_details = QLabel("Format: - | Resolution: -")
        self.meta_device = QLabel(f"Device: {self.engine.device.type.upper()}")
        self.meta_weights = QLabel(f"Weights: {os.path.basename(app_config.WEIGHTS_PATH)}")

        for lbl in [self.meta_details, self.meta_device, self.meta_weights]:
            lbl.setStyleSheet("color: #c7b4d0; font-size: 11px;")
            meta_box.addWidget(lbl)

        right_layout.addStretch()
        right_layout.addWidget(meta_frame)

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
        target_path = app_config.CURRENT_DIR+app_config.IMAGES_PATH

        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Galaxy Image",
            target_path,
            "FITS Files (*.fits *.fit *.fts)"
        )
        if file_path:
            self._load_image(file_path)

    def _open_coords_dialog(self):
        dialog = QDialog(self)
        dialog.setWindowTitle("Request Image")
        dialog.setMinimumWidth(320)
        dialog.setStyleSheet(
            "QDialog { background-color: #100d18; color: #c7b4d0; }"
            "QDialog QLabel, QDialog QLineEdit, QDialog QComboBox, "
            "QDialog QDialogButtonBox, QDialog QPushButton { color: #c7b4d0; }"
            "QDialog QLineEdit, QDialog QComboBox { background-color: #21182a; "
            "border: 1px solid #563763; border-radius: 4px; padding: 5px; }"
        )

        form = QFormLayout(dialog)
        form.setSpacing(10)

        label_style = "color: #c7b4d0; font-size: 12px;"

        def make_label(text):
            lbl = QLabel(text)
            lbl.setStyleSheet(label_style)
            return lbl

        ra_input = QLineEdit()
        dec_input = QLineEdit()
        size_input = QLineEdit("200.0")
        band_input = QComboBox()
        band_input.addItems(["g", "r", "z"])

        form.addRow(make_label("RA (degrees)"), ra_input)
        form.addRow(make_label("DEC (degrees)"), dec_input)
        form.addRow(make_label("Size (arcmin)"), size_input)
        form.addRow(make_label("Band"), band_input)

        buttons = QDialogButtonBox(
            QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel
        )
        buttons.accepted.connect(dialog.accept)
        buttons.rejected.connect(dialog.reject)
        form.addRow(buttons)

        if dialog.exec() != QDialog.DialogCode.Accepted:
            return

        try:
            ra = float(ra_input.text())
            dec = float(dec_input.text())
            arcmin_size = float(size_input.text())
        except ValueError:
            QMessageBox.warning(self, "Invalid Input", "RA, DEC and size must be numbers.")
            return
        band = band_input.currentText()

        try:
            file_path = get_fits_cutout(ra,dec,arcmin_size,band)
        except HTTPError:
            QMessageBox.warning(self, "Invalid Input", "RA, DEC and size must be within the footprint of Legacy Survey DR11.")
            return

        self._load_image(app_config.CURRENT_DIR+file_path)

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
                self.result_badge.setStyleSheet("background-color: #21182a; color: #c7b4d0; border: 1px solid #392a43;")#94a3b8
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
                    "background-color: #3b1d49; color: #f9e9ff; border: 1px solid #a33ac0; "
                    "font-weight: bold; font-size: 16px; border-radius: 8px; padding: 10px;"
                )
            else:
                self.result_badge.setText(f"ELLIPTICAL GALAXY ({res['confidence'] * 100:.1f}%)")
                self.result_badge.setStyleSheet(
                    "background-color: #482516; color: #fff1df; border: 1px solid #e66c12; "
                    "font-weight: bold; font-size: 16px; border-radius: 8px; padding: 10px;"
                )
        except Exception as e:
            QMessageBox.critical(self, "Prediction Error", str(e))
