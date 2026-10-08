import numpy as np
import pandas as pd
import sep
from astropy.io import fits
from matplotlib.patches import Ellipse

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

def arcsinh_stretch(image, scale=0.05):
    img_min, img_max = np.min(image), np.max(image)
    normalized = (image - img_min) / (img_max - img_min)
    stretched = np.arcsinh(normalized / scale) / np.arcsinh(1.0 / scale)
    return np.clip(stretched, 0, 1)


DIR = "C:/Users/16096/OneDrive - University of Helsinki/Bäcker, Tia M's files - Group project/"

fits_file_name = "cutout_359.0295_-0.9165.fits"

data = fits.open(DIR+"/fits/"+fits_file_name)[0].data
data = data.astype(data.dtype.newbyteorder('='))

objects, data_sub, bkg = background_object_extraction(data)
ellipses, subject_ellipse, subject_indx = get_ellipses(objects)
cleaned_data = clean_image(data_sub, bkg, objects, ellipses, subject_ellipse, subject_indx)

stretched_data = arcsinh_stretch(cleaned_data, scale=0.05)

# hdu = fits.PrimaryHDU(data=cleaned_data)
# hdu.writeto(DIR+'/cleaned_'+fits_file_name)


# hdu = fits.PrimaryHDU(data=stretched_data)
# hdu.writeto(DIR+'/stretched_'+fits_file_name)