import os
import numpy as np
from PIL import Image
from astropy.io import fits
import matplotlib.cm as cm
from matplotlib.patches import Ellipse
import sep
import torch
from PyQt6.QtGui import QImage
import app_config



def background_object_extraction(data):
    bkg = sep.Background(data)
    data_sub = data-bkg
    objects = sep.extract(data_sub, thresh=1.5, err=bkg.globalrms)
    return objects, data_sub, bkg

def get_nearest_point(array, point):
    A = np.array([_e.get_center() for _e in array])
    distances = np.linalg.norm(A-np.array(point), axis=1)
    min_index = np.argmin(distances)
    return min_index, array[min_index]

def get_ellipses(objects):
    ellipses = []

    for _i in range(len(objects)):
        _e = Ellipse(xy=(objects['x'][_i], objects['y'][_i]),
                    width=6*objects['a'][_i],
                    height=6*objects['b'][_i],
                    angle=objects['theta'][_i] * 180. / np.pi)
        _e.set_facecolor('none')
        _e.set_edgecolor('red')
        ellipses.append(_e)

    subject_indx, subject_ellipse = get_nearest_point(ellipses,[128,128])

    return ellipses, subject_ellipse, subject_indx

def clean_image(data_sub, bkg, objects, ellipses, subject_ellipse, subject_indx):
    non_subject_mask = np.zeros(data_sub.shape, dtype=bool)

    for _i in range(len(objects)):

        if _i == subject_indx:
            continue

        if any(subject_ellipse.contains_points(ellipses[_i].get_verts())):
            continue

        sep.mask_ellipse(
            non_subject_mask, 
            x=objects['x'][_i], 
            y=objects['y'][_i],
            a=objects['a'][_i], 
            b=objects['b'][_i], 
            theta=objects['theta'][_i], 
            r=4 
        )

    mean_bkg = bkg.globalback
    std_bkg = bkg.globalrms
    noise_fill = np.random.normal(loc=mean_bkg, scale=std_bkg, size=data_sub.shape)

    cleaned_data = data_sub.copy()
    cleaned_data[non_subject_mask] = noise_fill[non_subject_mask]

    return cleaned_data

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

    objects, data_sub, bkg = background_object_extraction(raw_data)
    ellipses, subject_ellipse, subject_indx = get_ellipses(objects)
    cleaned_data = clean_image(data_sub, bkg, objects, ellipses, subject_ellipse, subject_indx)

    stretched = arcsinh_stretch(cleaned_data, scale=app_config.STRETCH_SCALE)

    colored_rgba = (cm.inferno(stretched*5) * 255).astype(np.uint8)
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
