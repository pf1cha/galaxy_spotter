import torch
import numpy as np
import matplotlib.pyplot as plt
import config


def display_predictions_grid(model, dataset, device):
    model.eval()
    indices = np.random.choice(len(dataset), 9, replace=False)

    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
    fig.suptitle("Test Predictions vs Ground Truth (3x3 Grid)", fontsize=14, weight="bold")

    print("\n" + "=" * 65)
    print(f"{'Idx':<6}{'Ground Truth':<18}{'Prediction':<18}{'Confidence':<14}{'Match'}")
    print("-" * 65)

    with torch.no_grad():
        for idx, ax in zip(indices, axes.ravel()):
            img_tensor, true_label = dataset[idx]
            input_tensor = img_tensor.unsqueeze(0).to(device)

            output = model(input_tensor)
            prob = torch.sigmoid(output).item()
            pred_code = 1 if prob >= config.DECISION_THRESHOLD else 0
            conf = prob if pred_code == 1 else 1.0 - prob

            true_code = int(true_label.item())
            is_correct = (pred_code == true_code)

            status = "CORRECT" if is_correct else "WRONG"
            print(f"{idx:<6}{config.CLASS_NAMES[true_code]:<18}{config.CLASS_NAMES[pred_code]:<18}{conf * 100:>5.1f}%{'':<8}{status}")

            img = img_tensor.squeeze(0).cpu().numpy()
            ax.imshow(img, cmap="inferno", origin="lower")
            ax.axis("off")

            text_color = "darkgreen" if is_correct else "red"
            title_text = f"True: {config.CLASS_NAMES[true_code]}\nPred: {config.CLASS_NAMES[pred_code]} ({conf * 100:.1f}%)"
            ax.set_title(title_text, fontsize=10, color=text_color, weight="bold")

    print("=" * 65 + "\n")
    plt.tight_layout()
    plt.show()
