import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score


def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss = 0.0
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in loader:
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * images.size(0)
            probs = torch.sigmoid(outputs)
            preds = (probs >= 0.5).float()

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    loss = total_loss / len(loader.dataset)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)

    bal_acc = balanced_accuracy_score(all_labels, all_preds)

    e_mask = all_labels == 0
    s_mask = all_labels == 1

    elliptical_acc = accuracy_score(all_labels[e_mask], all_preds[e_mask])
    spiral_acc = accuracy_score(all_labels[s_mask], all_preds[s_mask])

    return loss, bal_acc, elliptical_acc, spiral_acc
