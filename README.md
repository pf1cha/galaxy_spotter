# Galaxy Spotter

Galaxy Spotter is a desktop application that classifies a galaxy as elliptical
or spiral from an astronomical FITS or jpg image. It combines a PyQt6 interface,
astronomical image preprocessing, and a convolutional neural network (CNN) implemented
with PyTorch.

The application can open a local FITS file or download a cutout from the Legacy
Survey DR11 by right ascension and declination. It then displays the image and
reports the predicted class together with the probability assigned to each class.

## Features

- Load local `.fits` images.
- Request a FITS cutout using sky coordinates, angular size, and the `g`, `r`, or
  `z` band.
- Remove background and reduce interference from nearby detected sources.
- Classify galaxies as elliptical or spiral.
- Display the confidence assigned to both classes.
- Run inference on a CUDA GPU when available, with automatic CPU fallback.

## Application workflow

```text
Select or request a galaxy image
               |
               v
      Process the image
               |
               v
   Classify it with the CNN
               |
               v
 View the prediction and confidence
```

The application handles image preparation and model inference automatically,
then presents the result in the desktop interface.

## CNN logic

The classifier is a binary convolutional neural network. Four convolution blocks
learn increasingly complex image features. Each block contains a `3 x 3`
convolution, batch normalization, ReLU activation, and `2 x 2` max pooling.

| Stage | Operation | Output channels |
| --- | --- | ---: |
| Input | Single-channel galaxy image | 1 |
| Block 1 | Conv, BatchNorm, ReLU, MaxPool | 32 |
| Block 2 | Conv, BatchNorm, ReLU, MaxPool | 64 |
| Block 3 | Conv, BatchNorm, ReLU, MaxPool | 128 |
| Block 4 | Conv, BatchNorm, ReLU, MaxPool | 256 |
| Pooling | Adaptive average pooling | 256 |
| Classifier | Dropout, Linear, ReLU, Dropout, Linear | 1 logit |

## Training pipeline

The example training code is located in `cnn_example/`. It:

1. Reads labels and FITS filenames from a CSV file.
2. Creates stratified training, validation, and test splits.
3. Normalizes each image and applies an arcsinh intensity stretch.
4. Trains with binary cross-entropy with logits and the AdamW optimizer.
5. Keeps the checkpoint with the lowest validation loss.
6. Stops early when validation loss no longer improves.
7. Reports balanced accuracy and accuracy for each galaxy class.
8. Exports the weights and preprocessing metadata for desktop inference.

Before training, set `CSV_PATH` and `FITS_DIR` in `cnn_example/config.py` to the
locations of your labels and FITS images. The CSV file is expected to contain at
least these columns:

| Column | Description |
| --- | --- |
| `filename` | Name of the FITS image |
| `class_code` | `0` for elliptical or `1` for spiral |

Run training from the training directory so its local imports resolve correctly:

```powershell
cd cnn_example
python train.py
```

## Installation

Python 3.10 or newer is recommended.

```powershell
git clone <repository-url>
cd <repository-directory>

py -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

On macOS or Linux, activate the environment with:

```bash
source .venv/bin/activate
```

## Running the application

The trained checkpoint must be available at:

```text
application/model_weights/best_galaxy_model.pth
```

Start the interface from the project root:

```powershell
python application/main.py
```

To classify a galaxy:

1. Select **Select FITS Image** and choose a local file, or select **Request by
   Coordinates** and enter the desired sky position.
2. Check that the image preview and file information are correct.
3. Select **Classify Galaxy**.
4. Review the predicted class and the elliptical/spiral confidence breakdown.

An internet connection is required only when requesting an image by coordinates.

## Packaging for Windows

Install PyInstaller and build a folder-based application from the project root:

```powershell
pip install pyinstaller

pyinstaller `
  --noconfirm `
  --clean `
  --windowed `
  --name GalaxySpotter `
  --paths application `
  --add-data "application\model_weights\best_galaxy_model.pth;model_weights" `
  --add-data "application\images;images" `
  --collect-all sep `
  application\main.py
```

The packaged program will be written to
`dist/GalaxySpotter/GalaxySpotter.exe`. Distribute the complete
`dist/GalaxySpotter` directory because the executable depends on the files stored
beside it.

## Project structure

```text
.
|-- application/
|   |-- images/              # Example and downloaded FITS images
|   |-- model_weights/       # Trained model checkpoints
|   |-- app_config.py        # Inference and display configuration
|   |-- galaxy_cnn.py        # CNN architecture
|   |-- image_loader.py      # FITS preprocessing and cutout download
|   |-- main.py              # Desktop application entry point
|   |-- predict.py           # Model loading and inference
|   |-- style.py             # PyQt6 stylesheet
|   `-- ui_window.py         # Main user interface
|-- cnn_example/
|   |-- config.py            # Training configuration
|   |-- dataset.py           # PyTorch FITS dataset
|   |-- evaluate.py          # Validation and test metrics
|   |-- galaxy_cnn.py        # Training copy of the architecture
|   |-- save_model.py        # Deployment bundle export
|   |-- train.py             # Training loop
|   `-- visualization.py     # Prediction visualization
|-- requirements.txt
`-- README.md
```

## Results

Add the metrics from your final test run here. Keeping this section tied to a
specific checkpoint and test split makes the results reproducible.

| Metric | Result |
| --- | ---: |
| Test loss | 0.3834 |
| Balanced accuracy | 83.97% |
| Elliptical accuracy | 85.45% |
| Spiral accuracy | 82.50% |
