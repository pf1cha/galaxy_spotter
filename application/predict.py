import os
import torch
from galaxy_cnn import GalaxyCNN
import app_config


class GalaxyInferenceEngine:
    def __init__(self):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = None
        self.is_ready = False
        self.error_message = ""
        self._initialize()

    def _initialize(self):
        target_path = app_config.CURRENT_DIR+app_config.WEIGHTS_PATH

        if not target_path or not os.path.isabs(target_path):
            self.error_message = f"Invalid or non-absolute path provided: '{target_path}'"
            self.is_ready = False
            return

        if not os.path.isfile(target_path):
            self.error_message = f"Weights file not found at: '{target_path}'"
            self.is_ready = False
            return

        try:
            self.model = GalaxyCNN(in_channels=1, num_classes=1)
            loaded_data = torch.load(target_path, map_location=self.device)

            if isinstance(loaded_data, dict) and "model_state_dict" in loaded_data:
                state_dict = loaded_data["model_state_dict"]
            else:
                state_dict = loaded_data

            self.model.load_state_dict(state_dict)
            self.model.to(self.device)
            self.model.eval()

            self.is_ready = True
            self.error_message = ""

        except Exception as e:
            self.model = None
            self.is_ready = False
            self.error_message = f"Corrupted or incompatible weights file: {str(e)}"

    def predict(self, tensor):
        if not self.is_ready or self.model is None:
            raise RuntimeError(f"Model is unavailable: {self.error_message}")

        with torch.no_grad():
            input_tensor = tensor.to(self.device)
            logit = self.model(input_tensor)
            prob_spiral = torch.sigmoid(logit).item()

        prob_elliptical = 1.0 - prob_spiral
        pred_code = 1 if prob_spiral >= app_config.DECISION_THRESHOLD else 0
        conf = prob_spiral if pred_code == 1 else prob_elliptical

        return {
            "pred_code": pred_code,
            "label": app_config.CLASS_NAMES[pred_code],
            "confidence": conf,
            "prob_spiral": prob_spiral,
            "prob_elliptical": prob_elliptical
        }