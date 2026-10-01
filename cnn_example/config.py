import os
import torch

CSV_PATH = "C:\\Users\\lonis\\OneDrive - University of Helsinki\\Файлы пользователя Bäcker, Tia M - Group project\\galaxy_data.csv"
FITS_DIR = "C:\\Users\\lonis\\OneDrive - University of Helsinki\\Файлы пользователя Bäcker, Tia M - Group project\\fits_preprocessed"

MODEL_SAVE_PATH = "best_galaxy_model.pth"
BUNDLE_SAVE_PATH = "galaxy_classifier_bundle.pt"

BATCH_SIZE = 32
NUM_EPOCHS = 5
LEARNING_RATE = 0.001
RANDOM_SEED = 239

TEST_SPLIT_SIZE = 0.15
VAL_SPLIT_SIZE = 0.15

# Change it here depending on the number of epoch in the training loop.
# With GPU I'd recommend more epochs, just to see how the model performs with more training.
EARLY_STOPPING_PATIENCE = 2
STRETCH_SCALE = 0.05
DECISION_THRESHOLD = 0.5

CLASS_NAMES = {0: "Elliptical", 1: "Spiral"}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HAS_CUDA = torch.cuda.is_available()