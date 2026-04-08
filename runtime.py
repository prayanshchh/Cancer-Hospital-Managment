import os
import sys
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

import joblib
import matplotlib
import numpy as np
import torch
import torch.nn as nn
from captum.attr import LayerAttribution, LayerGradCam
from PIL import Image
import torchvision.transforms as transforms
from transformers import AutoModel


matplotlib.use("Agg")
from matplotlib import cm  # noqa: E402


CLASS_NAMES = ["colon_aca", "colon_n", "lung_aca", "lung_n", "lung_scc"]
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
CONVNEXT_CHECKPOINT_NAME = "convnext_v2_epoch_30.pth"
PHIKON_CHECKPOINT_NAME = "phikon_best_overall.pth"
META_MODEL_NAME = "ultimate_patho_fusion_v1.pkl"

LABEL_METADATA = {
    "colon_aca": {
        "title": "Colon Adenocarcinoma",
        "group": "Cancerous tissue",
        "summary": "Malignant gland-forming tumor tissue from the colon.",
    },
    "colon_n": {
        "title": "Benign Colon Tissue",
        "group": "Normal tissue",
        "summary": "Non-cancerous colon histology.",
    },
    "lung_aca": {
        "title": "Lung Adenocarcinoma",
        "group": "Cancerous tissue",
        "summary": "A malignant epithelial tumor pattern from lung tissue.",
    },
    "lung_n": {
        "title": "Benign Lung Tissue",
        "group": "Normal tissue",
        "summary": "Non-cancerous lung histology.",
    },
    "lung_scc": {
        "title": "Lung Squamous Cell Carcinoma",
        "group": "Cancerous tissue",
        "summary": "Squamous malignancy pattern from lung tissue.",
    },
}


class StreamA_ConvNeXtV2(nn.Module):
    def __init__(self, num_classes: int = 5, local_files_only: bool = False):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            "facebook/convnextv2-base-22k-224",
            local_files_only=local_files_only,
        )
        self.head = nn.Sequential(
            nn.Linear(1024, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.base(x)
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            features = outputs.pooler_output
        else:
            features = outputs.last_hidden_state.mean(dim=1)
        return self.head(features)


class StreamB_Phikon(nn.Module):
    def __init__(self, num_classes: int = 5, local_files_only: bool = False):
        super().__init__()
        self.base = AutoModel.from_pretrained(
            "owkin/phikon",
            local_files_only=local_files_only,
        )
        for param in self.base.parameters():
            param.requires_grad = False
        self.head = nn.Sequential(
            nn.Linear(768, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        outputs = self.base(x)
        features = outputs.last_hidden_state[:, 0, :]
        return self.head(features)


class PathoFusionEnsemble:
    def __init__(self, model_a: Optional[nn.Module] = None, model_b: Optional[nn.Module] = None, meta_model: Any = None, device: Optional[torch.device] = None):
        self.model_a = model_a
        self.model_b = model_b
        self.meta_model = meta_model
        self.device = device
        self.class_names = CLASS_NAMES

    def predict(self, input_tensor: torch.Tensor) -> np.ndarray:
        self.model_a.eval()
        self.model_b.eval()

        with torch.no_grad():
            logits_a = self.model_a(input_tensor.to(self.device))
            probs_a = torch.softmax(logits_a, dim=1).cpu().numpy()

            logits_b = self.model_b(input_tensor.to(self.device))
            probs_b = torch.softmax(logits_b, dim=1).cpu().numpy()

            combined_features = np.hstack((probs_a, probs_b))
            return self.meta_model.predict(combined_features)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def use_local_hf_files() -> bool:
    value = os.getenv("HF_LOCAL_FILES_ONLY", "").strip().lower()
    return value in {"1", "true", "yes", "on"}


def register_pickle_classes() -> None:
    main_module = sys.modules["__main__"]
    setattr(main_module, "StreamA_ConvNeXtV2", StreamA_ConvNeXtV2)
    setattr(main_module, "StreamB_Phikon", StreamB_Phikon)
    setattr(main_module, "PathoFusionEnsemble", PathoFusionEnsemble)


def build_transform() -> transforms.Compose:
    return transforms.Compose(
        [
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.ToTensor(),
            transforms.Normalize(mean=MEAN.tolist(), std=STD.tolist()),
        ]
    )


@lru_cache(maxsize=1)
def load_runtime_bundle(
    convnext_weights: str,
    phikon_weights: str,
    meta_model_path: str,
    local_files_only: Optional[bool] = None,
) -> Tuple[StreamA_ConvNeXtV2, StreamB_Phikon, Any, torch.device]:
    device = get_device()
    if local_files_only is None:
        local_files_only = use_local_hf_files()

    if not os.path.exists(convnext_weights):
        raise FileNotFoundError(
            f"Could not find ConvNeXt checkpoint at '{convnext_weights}'. "
            f"Make sure '{CONVNEXT_CHECKPOINT_NAME}' is present in the project root."
        )
    if not os.path.exists(phikon_weights):
        raise FileNotFoundError(
            f"Could not find Phikon checkpoint at '{phikon_weights}'. "
            f"Make sure '{PHIKON_CHECKPOINT_NAME}' is present in the project root."
        )
    if not os.path.exists(meta_model_path):
        raise FileNotFoundError(
            f"Could not find fusion model at '{meta_model_path}'. "
            f"Make sure '{META_MODEL_NAME}' is present in the project root."
        )

    model_a = StreamA_ConvNeXtV2(num_classes=len(CLASS_NAMES), local_files_only=local_files_only)
    model_a.load_state_dict(torch.load(convnext_weights, map_location=device))
    model_a = model_a.to(device).eval()

    model_b = StreamB_Phikon(num_classes=len(CLASS_NAMES), local_files_only=local_files_only)
    model_b.load_state_dict(torch.load(phikon_weights, map_location=device), strict=False)
    model_b = model_b.to(device).eval()

    register_pickle_classes()
    meta_model = joblib.load(meta_model_path)
    return model_a, model_b, meta_model, device


@lru_cache(maxsize=1)
def load_convnext_model(
    convnext_weights: str,
    local_files_only: Optional[bool] = None,
) -> Tuple[StreamA_ConvNeXtV2, torch.device]:
    device = get_device()
    if local_files_only is None:
        local_files_only = use_local_hf_files()
    if not os.path.exists(convnext_weights):
        raise FileNotFoundError(
            f"Could not find ConvNeXt checkpoint at '{convnext_weights}'. "
            f"Make sure '{CONVNEXT_CHECKPOINT_NAME}' is present in the project root."
        )

    model = StreamA_ConvNeXtV2(num_classes=len(CLASS_NAMES), local_files_only=local_files_only)
    model.load_state_dict(torch.load(convnext_weights, map_location=device))
    model = model.to(device).eval()
    return model, device


def unwrap_meta_model(meta_model: Any) -> Any:
    if hasattr(meta_model, "meta_model"):
        return meta_model.meta_model
    return meta_model


def preprocess_image(image: Image.Image) -> torch.Tensor:
    return build_transform()(image.convert("RGB")).unsqueeze(0)


def denormalize_image(tensor: torch.Tensor) -> np.ndarray:
    array = tensor.squeeze(0).detach().cpu().numpy()
    array = np.transpose(array, (1, 2, 0))
    array = np.clip(array * STD + MEAN, 0.0, 1.0)
    return array


def is_container_module(module: nn.Module) -> bool:
    return isinstance(module, (nn.ModuleList, nn.Sequential, nn.ModuleDict))


def resolve_leaf_tensor_layer(module: Optional[nn.Module]) -> Optional[nn.Module]:
    if module is None:
        return None

    if isinstance(module, nn.ModuleList):
        for child in reversed(list(module)):
            resolved = resolve_leaf_tensor_layer(child)
            if resolved is not None:
                return resolved
        return None

    children = list(module.children())
    if not children and not is_container_module(module):
        return module

    for child in reversed(children):
        resolved = resolve_leaf_tensor_layer(child)
        if resolved is not None:
            return resolved

    if not is_container_module(module):
        return module

    return None


def get_last_tensor_layer(module: nn.Module) -> nn.Module:
    conv_candidates: List[nn.Module] = []
    fallback_candidates: List[nn.Module] = []

    if hasattr(module, "base"):
        base = module.base

        if hasattr(base, "encoder") and hasattr(base.encoder, "stages"):
            try:
                last_block = base.encoder.stages[-1].layers[-1]
                fallback_candidates.append(last_block)
                if hasattr(last_block, "dwconv"):
                    conv_candidates.append(last_block.dwconv)
            except Exception:
                pass
            try:
                fallback_candidates.append(base.encoder.stages[-1])
            except Exception:
                pass

        if hasattr(base, "features"):
            try:
                fallback_candidates.append(base.features[-1])
            except Exception:
                pass

    for layer in reversed(list(module.modules())):
        if isinstance(layer, torch.nn.Conv2d):
            conv_candidates.append(layer)
        elif isinstance(layer, (torch.nn.BatchNorm2d, torch.nn.LayerNorm)):
            fallback_candidates.append(layer)

    for candidate in conv_candidates + fallback_candidates:
        resolved = resolve_leaf_tensor_layer(candidate)
        if resolved is not None and not is_container_module(resolved):
            return resolved

    raise ValueError("Could not find a suitable tensor-producing layer for Grad-CAM.")


def softmax_row(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values, axis=1, keepdims=True)
    exp_values = np.exp(shifted)
    return exp_values / np.sum(exp_values, axis=1, keepdims=True)


def compute_probabilities(
    model_a: nn.Module,
    model_b: nn.Module,
    meta_model: Any,
    device: torch.device,
    input_tensor: torch.Tensor,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    with torch.inference_mode():
        logits_a = model_a(input_tensor.to(device))
        probs_a = torch.softmax(logits_a, dim=1).cpu().numpy()

        logits_b = model_b(input_tensor.to(device))
        probs_b = torch.softmax(logits_b, dim=1).cpu().numpy()

    combined_features = np.hstack((probs_a, probs_b))
    actual_model = unwrap_meta_model(meta_model)

    if hasattr(actual_model, "predict_proba"):
        fused_probs = actual_model.predict_proba(combined_features)
    elif hasattr(actual_model, "decision_function"):
        fused_probs = softmax_row(actual_model.decision_function(combined_features))
    else:
        fused_pred = actual_model.predict(combined_features)
        fused_probs = np.zeros((combined_features.shape[0], len(CLASS_NAMES)), dtype=np.float32)
        for row_idx, class_idx in enumerate(fused_pred):
            fused_probs[row_idx, int(class_idx)] = 1.0

    return probs_a[0], probs_b[0], fused_probs[0]


def class_display_name(class_key: str) -> str:
    return LABEL_METADATA[class_key]["title"]


def build_class_rows(convnext_probs: np.ndarray, phikon_probs: np.ndarray, fused_probs: np.ndarray) -> List[Dict[str, Any]]:
    rows = []
    for idx, class_key in enumerate(CLASS_NAMES):
        rows.append(
            {
                "class_index": idx,
                "class_key": class_key,
                "label": class_display_name(class_key),
                "group": LABEL_METADATA[class_key]["group"],
                "convnext_confidence": round(float(convnext_probs[idx]) * 100, 2),
                "phikon_confidence": round(float(phikon_probs[idx]) * 100, 2),
                "fusion_confidence": round(float(fused_probs[idx]) * 100, 2),
            }
        )
    return rows


def predict_image(
    image: Image.Image,
    convnext_weights: str,
    phikon_weights: str,
    meta_model_path: str,
) -> Dict[str, Any]:
    model_a, model_b, meta_model, device = load_runtime_bundle(
        convnext_weights=convnext_weights,
        phikon_weights=phikon_weights,
        meta_model_path=meta_model_path,
    )
    input_tensor = preprocess_image(image)
    convnext_probs, phikon_probs, fused_probs = compute_probabilities(
        model_a=model_a,
        model_b=model_b,
        meta_model=meta_model,
        device=device,
        input_tensor=input_tensor,
    )
    actual_model = unwrap_meta_model(meta_model)
    meta_features = np.hstack((convnext_probs.reshape(1, -1), phikon_probs.reshape(1, -1)))
    pred_idx = int(actual_model.predict(meta_features)[0])
    pred_key = CLASS_NAMES[pred_idx]

    ranked_indices = np.argsort(fused_probs)[::-1]
    ranked_predictions = [
        {
            "class_key": CLASS_NAMES[idx],
            "label": class_display_name(CLASS_NAMES[idx]),
            "confidence": round(float(fused_probs[idx]) * 100, 2),
            "group": LABEL_METADATA[CLASS_NAMES[idx]]["group"],
        }
        for idx in ranked_indices[:3]
    ]

    return {
        "prediction": {
            "class_index": pred_idx,
            "class_key": pred_key,
            "label": class_display_name(pred_key),
            "group": LABEL_METADATA[pred_key]["group"],
            "summary": LABEL_METADATA[pred_key]["summary"],
            "confidence": round(float(fused_probs[pred_idx]) * 100, 2),
        },
        "class_probabilities": build_class_rows(convnext_probs, phikon_probs, fused_probs),
        "model_breakdown": {
            "convnext": round(float(convnext_probs[pred_idx]) * 100, 2),
            "phikon": round(float(phikon_probs[pred_idx]) * 100, 2),
            "fusion": round(float(fused_probs[pred_idx]) * 100, 2),
        },
        "top_predictions": ranked_predictions,
    }


def overlay_heatmap(image_array: np.ndarray, heatmap: np.ndarray) -> Image.Image:
    heatmap = np.clip(heatmap, 0.0, 1.0)
    colored = cm.get_cmap("turbo")(heatmap)[..., :3]
    overlay = (0.5 * image_array) + (0.5 * colored)
    overlay = np.clip(overlay, 0.0, 1.0)
    overlay_uint8 = (overlay * 255).astype(np.uint8)
    return Image.fromarray(overlay_uint8)


def build_gradcam_explanation(
    image: Image.Image,
    convnext_checkpoint_path: str,
    target_idx: Optional[int] = None,
) -> Dict[str, Any]:
    model, device = load_convnext_model(convnext_checkpoint_path)
    input_tensor = preprocess_image(image).to(device)
    input_tensor.requires_grad_(True)
    with torch.inference_mode():
        logits = model(input_tensor)
        convnext_probs = torch.softmax(logits, dim=1).detach().cpu().numpy()[0]
        if target_idx is None:
            target_idx = int(np.argmax(convnext_probs))

    target_layer = get_last_tensor_layer(model)
    layer_gc = LayerGradCam(model, target_layer)
    gc_attr = layer_gc.attribute(input_tensor, target=target_idx)
    gc_attr = LayerAttribution.interpolate(gc_attr, input_tensor.shape[2:])
    heatmap = gc_attr.squeeze().detach().cpu().numpy()
    heatmap = np.maximum(heatmap, 0)
    heatmap = heatmap / (np.max(heatmap) + 1e-8)

    original = denormalize_image(input_tensor)
    overlay = overlay_heatmap(original, heatmap)

    return {
        "overlay": overlay,
        "predicted_class": CLASS_NAMES[target_idx],
        "convnext_confidence": round(float(convnext_probs[target_idx]) * 100, 2),
    }
