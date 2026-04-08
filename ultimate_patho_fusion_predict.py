import argparse
import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import torch
import torch.nn as nn
import torchvision.transforms as transforms
from transformers import AutoModel

CLASS_NAMES = ['colon_aca', 'colon_n', 'lung_aca', 'lung_n', 'lung_scc']


def get_device(force_cpu=False):
    if force_cpu:
        return torch.device("cpu")
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


class StreamA_ConvNeXtV2(nn.Module):
    def __init__(self, num_classes=5, local_files_only=False):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            "facebook/convnextv2-base-22k-224",
            local_files_only=local_files_only
        )
        # Must match training-time architecture exactly
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.base(x)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        return self.head(features)


class StreamB_Phikon(nn.Module):
    def __init__(self, num_classes=5, local_files_only=False):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            "owkin/phikon",
            local_files_only=local_files_only
        )

        for param in self.base.parameters():
            param.requires_grad = False

        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        outputs = self.base(x)
        features = outputs.last_hidden_state[:, 0, :]
        return self.head(features)


# IMPORTANT:
# This class must exist so joblib can unpickle ultimate_patho_fusion_v1.pkl
class PathoFusionEnsemble:
    def __init__(self, model_a=None, model_b=None, meta_model=None, device=None):
        self.model_a = model_a
        self.model_b = model_b
        self.meta_model = meta_model
        self.device = device
        self.class_names = CLASS_NAMES

    def predict(self, input_tensor):
        self.model_a.eval()
        self.model_b.eval()

        with torch.no_grad():
            logits_a = self.model_a(input_tensor.to(self.device))
            probs_a = torch.softmax(logits_a, dim=1).cpu().numpy()

            logits_b = self.model_b(input_tensor.to(self.device))
            probs_b = torch.softmax(logits_b, dim=1).cpu().numpy()

        combined_features = np.hstack((probs_a, probs_b))

        if hasattr(self.meta_model, "predict"):
            return self.meta_model.predict(combined_features)

        raise AttributeError("Loaded PathoFusionEnsemble has no valid meta_model.predict")


def build_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])


def load_models(device, convnext_weights, phikon_weights, meta_model_path, local_files_only):
    print(f"Using device: {device}")

    print("Loading Stream A...")
    model_a = StreamA_ConvNeXtV2(local_files_only=local_files_only)
    model_a.load_state_dict(torch.load(convnext_weights, map_location=device))
    model_a = model_a.to(device).eval()

    print("Loading Stream B...")
    model_b = StreamB_Phikon(local_files_only=local_files_only)
    model_b.load_state_dict(torch.load(phikon_weights, map_location=device), strict=False)
    model_b = model_b.to(device).eval()

    print("Loading Fusion Model...")
    meta_model = joblib.load(meta_model_path)

    return model_a, model_b, meta_model


def unwrap_meta_model(meta_model):
    # If the pickle contains the wrapper object, use its internal sklearn model
    if hasattr(meta_model, "meta_model"):
        return meta_model.meta_model
    return meta_model


def predict_single_image(image_path, model_a, model_b, meta_model, device):
    img = Image.open(image_path).convert("RGB")
    x = build_transform()(img).unsqueeze(0).to(device)

    with torch.inference_mode():
        probs_a = torch.softmax(model_a(x), dim=1).cpu().numpy()  # (1,5)
        probs_b = torch.softmax(model_b(x), dim=1).cpu().numpy()  # (1,5)

    meta_features = np.hstack((probs_a, probs_b))  # (1,10)

    actual_model = unwrap_meta_model(meta_model)

    pred_arr = actual_model.predict(meta_features)
    pred_idx = int(pred_arr[0])

    if hasattr(actual_model, "predict_proba"):
        fused_probs = actual_model.predict_proba(meta_features)[0]
    else:
        fused_probs = np.zeros(len(CLASS_NAMES), dtype=float)
        fused_probs[pred_idx] = 1.0

    result = {
        "image_path": str(Path(image_path).resolve()),
        "predicted_class": CLASS_NAMES[pred_idx],
        "predicted_index": pred_idx,
        "confidence": float(fused_probs[pred_idx]),
        "fused": fused_probs.tolist(),
        "stream_a": probs_a[0].tolist(),
        "stream_b": probs_b[0].tolist()
    }

    return img, result


def save_report(img, result, path):
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    axes[0].imshow(img)
    axes[0].axis("off")
    axes[0].set_title("Input Image")

    y = np.arange(len(CLASS_NAMES))

    axes[1].barh(y - 0.2, result["stream_a"], 0.2, label="ConvNeXt")
    axes[1].barh(y, result["stream_b"], 0.2, label="Phikon")
    axes[1].barh(y + 0.2, result["fused"], 0.2, label="Fusion")

    axes[1].set_yticks(y)
    axes[1].set_yticklabels(CLASS_NAMES)
    axes[1].set_xlim(0, 1)
    axes[1].set_xlabel("Probability")
    axes[1].set_title(f"Prediction: {result['predicted_class']} ({result['confidence']:.4f})")
    axes[1].legend()

    plt.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, dpi=180, bbox_inches="tight")
    plt.close()


def save_json(result, path):
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--convnext-weights", default="convnext_v2_epoch_30.pth")
    parser.add_argument("--phikon-weights", default="phikon_best_overall.pth")
    parser.add_argument("--meta-model", default="ultimate_patho_fusion_v1.pkl")
    parser.add_argument("--output", default="outputs/report.png")
    parser.add_argument("--output-json", default="outputs/report.json")
    parser.add_argument("--cpu", action="store_true")
    parser.add_argument("--local-files-only", action="store_true")
    args = parser.parse_args()

    for p, label in [
        (args.image, "Input image"),
        (args.convnext_weights, "ConvNeXt weights"),
        (args.phikon_weights, "Phikon weights"),
        (args.meta_model, "Fusion model"),
    ]:
        if not Path(p).exists():
            raise FileNotFoundError(f"{label} not found: {p}")

    device = get_device(args.cpu)

    model_a, model_b, meta_model = load_models(
        device,
        args.convnext_weights,
        args.phikon_weights,
        args.meta_model,
        args.local_files_only
    )

    img, result = predict_single_image(
        args.image,
        model_a,
        model_b,
        meta_model,
        device
    )

    save_report(img, result, args.output)
    save_json(result, args.output_json)

    print("\nPrediction complete")
    print("-" * 50)
    print("Predicted class :", result["predicted_class"])
    print("Confidence      :", result["confidence"])
    print("Report image    :", args.output)
    print("JSON report     :", args.output_json)


if __name__ == "__main__":
    main()
