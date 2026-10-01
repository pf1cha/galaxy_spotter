import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from tqdm import tqdm

import config
from dataset import GalaxyDataset
from galaxy_cnn import GalaxyCNN
from evaluate import evaluate
from visualization import display_predictions_grid
from save_model import save_model_package


def train_epoch(model, loader, criterion, optimizer, device, epoch, total_epochs):
    model.train()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    progress_bar = tqdm(loader, desc=f"Epoch {epoch:02d}/{total_epochs:02d} [Training]", unit="batch", leave=False)

    for images, labels in progress_bar:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        probs = torch.sigmoid(outputs)
        preds = (probs >= config.DECISION_THRESHOLD).float()

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        current_loss = total_loss / len(all_labels)
        current_acc = accuracy_score(all_labels, all_preds) * 100
        progress_bar.set_postfix({"loss": f"{current_loss:.4f}", "acc": f"{current_acc:.2f}%"})

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = accuracy_score(all_labels, all_preds)
    return epoch_loss, epoch_acc


def run_pipeline():
    torch.manual_seed(config.RANDOM_SEED)
    np.random.seed(config.RANDOM_SEED)

    df = pd.read_csv(config.CSV_PATH)

    train_df, test_df = train_test_split(
        df, test_size=config.TEST_SPLIT_SIZE, random_state=config.RANDOM_SEED, stratify=df['class_code']
    )
    train_df, val_df = train_test_split(
        train_df, test_size=config.VAL_SPLIT_SIZE, random_state=config.RANDOM_SEED, stratify=train_df['class_code']
    )

    train_dataset = GalaxyDataset(train_df, config.FITS_DIR)
    val_dataset = GalaxyDataset(val_df, config.FITS_DIR)
    test_dataset = GalaxyDataset(test_df, config.FITS_DIR)

    train_loader = DataLoader(train_dataset, batch_size=config.BATCH_SIZE, shuffle=True, num_workers=0, pin_memory=config.HAS_CUDA)
    val_loader = DataLoader(val_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=config.HAS_CUDA)
    test_loader = DataLoader(test_dataset, batch_size=config.BATCH_SIZE, shuffle=False, num_workers=0, pin_memory=config.HAS_CUDA)

    model = GalaxyCNN(in_channels=1, num_classes=1).to(config.DEVICE)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=config.LEARNING_RATE, weight_decay=1e-4)

    best_val_loss = float('inf')
    epochs_no_improve = 0

    header = f"{'Epoch':^8} | {'Train Loss':^10} | {'Train Acc':^10} | {'Val Loss':^10} | {'Val Bal Acc':^12} | {'Val E-Acc':^11} | {'Val S-Acc':^11}"
    separator = "-" * len(header)
    print("\n" + "=" * len(header))
    print(header)
    print("=" * len(header))

    for epoch in range(1, config.NUM_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, config.DEVICE, epoch, config.NUM_EPOCHS)
        val_loss, val_bacc, val_e_acc, val_s_acc = evaluate(model, val_loader, criterion, config.DEVICE)

        is_best = val_loss < best_val_loss

        if is_best:
            best_val_loss = val_loss
            epochs_no_improve = 0
            torch.save(model.state_dict(), config.MODEL_SAVE_PATH)
            marker = " *"
        else:
            epochs_no_improve += 1
            marker = f" ({epochs_no_improve}/{config.EARLY_STOPPING_PATIENCE})"

        print(f" {epoch:02d}/{config.NUM_EPOCHS:02d}   |  {train_loss:8.4f}  |  {train_acc * 100:7.2f}%  |  {val_loss:8.4f}  |  {val_bacc * 100:9.2f}%  |  {val_e_acc * 100:8.2f}%  |  {val_s_acc * 100:8.2f}% {marker}")

        if epochs_no_improve >= config.EARLY_STOPPING_PATIENCE:
            print(separator)
            print(f"\n[Early Stopping Triggered] Validation loss did not improve for {config.EARLY_STOPPING_PATIENCE} consecutive epochs.")
            print(f"Terminating training early at epoch {epoch}.")
            break

    print(separator)
    print("(*) Indicates new best checkpoint saved.\n")

    model.load_state_dict(torch.load(config.MODEL_SAVE_PATH, map_location=config.DEVICE))
    test_loss, test_bacc, test_e_acc, test_s_acc = evaluate(model, test_loader, criterion, config.DEVICE)

    print("=" * 45)
    print(f"{'Final Test Set Evaluation':^45}")
    print("=" * 45)
    print(f"  Cross-Entropy Loss:       {test_loss:.4f}")
    print(f"  Balanced Accuracy:        {test_bacc * 100:.2f}%")
    print(f"  Elliptical Accuracy (E):  {test_e_acc * 100:.2f}%")
    print(f"  Spiral Accuracy (S):      {test_s_acc * 100:.2f}%")
    print("=" * 45)

    save_model_package(
        model=model,
        filepath=config.BUNDLE_SAVE_PATH,
        class_names=config.CLASS_NAMES,
        stretch_scale=config.STRETCH_SCALE,
        threshold=config.DECISION_THRESHOLD
    )

    display_predictions_grid(model, test_dataset, config.DEVICE)


if __name__ == "__main__":
    run_pipeline()
