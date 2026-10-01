import torch


def save_model_package(model, filepath, class_names, stretch_scale=0.05, threshold=0.5):
    bundle = {
        "model_state_dict": model.state_dict(),
        "model_kwargs": {
            "in_channels": 1,
            "num_classes": 1
        },
        "preprocessing": {
            "stretch_scale": stretch_scale,
            "nan_fill": 0.0,
            "clip_min": 0.0,
            "clip_max": 1.0
        },
        "metadata": {
            "class_names": class_names,
            "threshold": threshold
        }
    }
    torch.save(bundle, filepath)
    print(f"[Export] Complete desktop deployment bundle saved to: '{filepath}'")
