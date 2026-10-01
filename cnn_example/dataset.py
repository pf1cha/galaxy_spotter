import os
import numpy as np
import torch
from torch.utils.data import Dataset
from astropy.io import fits
import config


def arcsinh_stretch(image, scale=config.STRETCH_SCALE):
    image = np.nan_to_num(image, nan=0.0, posinf=0.0, neginf=0.0)
    img_min = np.min(image)
    img_max = np.max(image)
    if img_max > img_min:
        normalized = (image - img_min) / (img_max - img_min)
    else:
        normalized = np.zeros_like(image)
    stretched = np.arcsinh(normalized / scale) / np.arcsinh(1.0 / scale)
    return np.clip(stretched, 0.0, 1.0).astype(np.float32)


class GalaxyDataset(Dataset):
    def __init__(self, df, img_dir):
        self.df = df.reset_index(drop=True)
        self.img_dir = img_dir

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        file_path = os.path.join(self.img_dir, str(row['filename']))

        with fits.open(file_path, memmap=False) as hdul:
            data = hdul[0].data

        data = np.array(data, dtype=np.float32)
        data = arcsinh_stretch(data, scale=config.STRETCH_SCALE)

        tensor = torch.from_numpy(data.copy()).unsqueeze(0)
        label = torch.tensor(row['class_code'], dtype=torch.float32)
        return tensor, label