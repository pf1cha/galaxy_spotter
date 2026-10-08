import os
import numpy as np
from PIL import Image
from astropy.io import fits
import matplotlib.cm as cm
import torch
from PyQt6.QtGui import QImage
import app_config


def arcsinh_stretch(image, scale=app_config.STRETCH_SCALE):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = np.zeros_like(image)
    stretched = np.arcsinh(normalized / scale) / np.arcsinh(1.0 / scale)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


def process_fits(file_path):
    with fits.open(file_path, memmap=False) as hdul:
        raw_data = None
        for hdu in hdul:
            if hdu.data is not None and hdu.data.ndim >= 2:
                raw_data = hdu.data
                break
        if raw_data is None:
            raise ValueError("No valid 2D image data array found in FITS extensions.")

    if raw_data.ndim == 3:
        raw_data = raw_data[0]

    raw_data = np.array(raw_data, dtype=np.float32)
    stretched = arcsinh_stretch(raw_data, scale=app_config.STRETCH_SCALE)

    colored_rgba = (cm.inferno(stretched) * 255).astype(np.uint8)
    h, w, _ = colored_rgba.shape
    qimg = QImage(colored_rgba.data, w, h, 4 * w, QImage.Format.Format_RGBA8888)

    tensor = torch.from_numpy(stretched).unsqueeze(0).unsqueeze(0)
    info_text = f"Format: FITS CCD | Resolution: {w} × {h}"
    return tensor, qimg, info_text


def process_optical(file_path):
    ext = os.path.splitext(file_path)[1].upper().replace(".", "")
    img = Image.open(file_path)
    w, h = img.size

    gray = img.convert("L")
    arr_gray = np.array(gray, dtype=np.float32) / 255.0
    stretched = arcsinh_stretch(arr_gray, scale=app_config.STRETCH_SCALE)

    rgb = img.convert("RGB")
    arr_rgb = np.array(rgb, dtype=np.uint8)
    qimg = QImage(arr_rgb.data, w, h, 3 * w, QImage.Format.Format_RGB888)

    tensor = torch.from_numpy(stretched).unsqueeze(0).unsqueeze(0)
    info_text = f"Format: {ext} Optical | Resolution: {w} × {h}"
    return tensor, qimg, info_text


def load_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext in [".fits", ".fit", ".fts"]:
        return process_fits(file_path)
    return process_optical(file_path)
