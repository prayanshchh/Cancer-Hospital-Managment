import argparse
import base64
import io
import os
import time
from typing import Any, Dict, Tuple

from flask import Flask, Response, g, jsonify, request
from PIL import Image
from prometheus_client import CONTENT_TYPE_LATEST, Counter, Gauge, Histogram, generate_latest

from runtime import (
    CONVNEXT_CHECKPOINT_NAME,
    META_MODEL_NAME,
    PHIKON_CHECKPOINT_NAME,
    build_gradcam_explanation,
    get_device,
    load_convnext_model,
    load_runtime_bundle,
    predict_image,
)


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, META_MODEL_NAME)
CONVNEXT_PATH = os.path.join(BASE_DIR, CONVNEXT_CHECKPOINT_NAME)
PHIKON_PATH = os.path.join(BASE_DIR, PHIKON_CHECKPOINT_NAME)

APP_NAME = "Cancer Pathology Model API"

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024

METRICS_EXCLUDED_PATH_PREFIXES = ("/metrics",)

HTTP_REQUESTS_TOTAL = Counter(
    "cancer_model_api_http_requests_total",
    "Total HTTP requests handled by the model API.",
    ["method", "route", "status_code"],
)
HTTP_REQUEST_DURATION_SECONDS = Histogram(
    "cancer_model_api_http_request_duration_seconds",
    "Latency of model API HTTP requests.",
    ["method", "route"],
    buckets=(0.05, 0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60),
)
INFERENCE_TOTAL = Counter(
    "cancer_model_api_inference_total",
    "Number of model API inference stages completed.",
    ["stage", "status"],
)
INFERENCE_DURATION_SECONDS = Histogram(
    "cancer_model_api_inference_duration_seconds",
    "Latency of model API prediction and Grad-CAM stages.",
    ["stage"],
    buckets=(0.1, 0.25, 0.5, 1, 2, 5, 10, 30, 60, 120, 300),
)
ARTIFACT_READY = Gauge(
    "cancer_model_api_artifact_ready",
    "Whether required model artifacts are present on the host.",
    ["artifact"],
)
DEVICE_AVAILABLE = Gauge(
    "cancer_model_api_device_available",
    "Whether the host model API has a backend device available.",
    ["backend"],
)


def pil_to_data_url(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)
    encoded = base64.b64encode(buffer.getvalue()).decode("utf-8")
    return f"data:image/{fmt.lower()};base64,{encoded}"


def build_artifact_status() -> Dict[str, bool]:
    return {
        "fusion": os.path.exists(MODEL_PATH),
        "convnext": os.path.exists(CONVNEXT_PATH),
        "phikon": os.path.exists(PHIKON_PATH),
    }


def should_skip_metrics(path: str) -> bool:
    return any(path.startswith(prefix) for prefix in METRICS_EXCLUDED_PATH_PREFIXES)


def initialize_metrics() -> None:
    for rule in app.url_map.iter_rules():
        methods = sorted(method for method in rule.methods if method in {"GET", "POST"})
        for method in methods:
            HTTP_REQUESTS_TOTAL.labels(method=method, route=rule.rule, status_code="200")
            HTTP_REQUEST_DURATION_SECONDS.labels(method=method, route=rule.rule)

    for stage in ("prediction", "gradcam"):
        for status in ("success", "error"):
            INFERENCE_TOTAL.labels(stage=stage, status=status)
    for stage in ("prediction", "gradcam"):
        INFERENCE_DURATION_SECONDS.labels(stage=stage)

    for artifact in ("fusion", "convnext", "phikon"):
        ARTIFACT_READY.labels(artifact=artifact)
    for backend in ("cuda", "mps", "cpu"):
        DEVICE_AVAILABLE.labels(backend=backend)


def update_runtime_metrics() -> Dict[str, Any]:
    artifacts = build_artifact_status()
    for artifact, ready in artifacts.items():
        ARTIFACT_READY.labels(artifact=artifact).set(1 if ready else 0)

    device = get_device()
    DEVICE_AVAILABLE.labels(backend="cuda").set(1 if device.type == "cuda" else 0)
    DEVICE_AVAILABLE.labels(backend="mps").set(1 if device.type == "mps" else 0)
    DEVICE_AVAILABLE.labels(backend="cpu").set(1 if device.type == "cpu" else 0)
    return {"artifacts": artifacts, "device": device}


def decode_request_image() -> Tuple[Image.Image, str, bool]:
    if request.is_json:
        payload = request.get_json(silent=True) or {}
        image_base64 = payload.get("image_base64", "")
        if not image_base64:
            raise ValueError("JSON body must include image_base64.")
        image_bytes = base64.b64decode(image_base64)
        filename = str(payload.get("filename") or "upload.png")
        include_gradcam = bool(payload.get("include_gradcam", True))
        return Image.open(io.BytesIO(image_bytes)).convert("RGB"), filename, include_gradcam

    uploaded = request.files.get("image")
    if uploaded is None or uploaded.filename == "":
        raise ValueError("Request must include an image file.")

    image_bytes = uploaded.read()
    filename = uploaded.filename or "upload.png"
    include_gradcam = request.form.get("include_gradcam", "true").strip().lower() not in {"0", "false", "no"}
    return Image.open(io.BytesIO(image_bytes)).convert("RGB"), filename, include_gradcam


@app.before_request
def start_request_timer() -> None:
    if should_skip_metrics(request.path):
        return
    g.request_started_at = time.perf_counter()


@app.after_request
def observe_request_metrics(response: Response) -> Response:
    if should_skip_metrics(request.path):
        return response

    started_at = getattr(g, "request_started_at", None)
    if started_at is None:
        return response

    route = request.url_rule.rule if request.url_rule is not None and request.url_rule.rule else request.path
    duration = time.perf_counter() - started_at
    HTTP_REQUESTS_TOTAL.labels(
        method=request.method,
        route=route,
        status_code=str(response.status_code),
    ).inc()
    HTTP_REQUEST_DURATION_SECONDS.labels(
        method=request.method,
        route=route,
    ).observe(duration)
    return response


@app.route("/healthz")
def healthz():
    runtime = update_runtime_metrics()
    device = runtime["device"]
    return jsonify(
        {
            "status": "ok",
            "service": APP_NAME,
            "artifacts": runtime["artifacts"],
            "device": {
                "type": device.type,
                "cuda": device.type == "cuda",
                "mps": device.type == "mps",
                "cpu": device.type == "cpu",
                "cuda_device_count": 0,
            },
            "cache": {
                "fusion_bundle_loaded": load_runtime_bundle.cache_info().currsize > 0,
                "convnext_loaded": load_convnext_model.cache_info().currsize > 0,
            },
        }
    )


@app.route("/metrics")
def metrics() -> Response:
    update_runtime_metrics()
    return Response(generate_latest(), mimetype=CONTENT_TYPE_LATEST)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        image, filename, include_gradcam = decode_request_image()
    except Exception as exc:
        return jsonify({"status": "error", "error": str(exc)}), 400

    try:
        prediction_started_at = time.perf_counter()
        prediction = predict_image(
            image=image,
            convnext_weights=CONVNEXT_PATH,
            phikon_weights=PHIKON_PATH,
            meta_model_path=MODEL_PATH,
        )
        prediction_duration_seconds = time.perf_counter() - prediction_started_at
        INFERENCE_TOTAL.labels(stage="prediction", status="success").inc()
        INFERENCE_DURATION_SECONDS.labels(stage="prediction").observe(prediction_duration_seconds)

        response_payload: Dict[str, Any] = {
            **prediction,
            "status": "ok",
            "filename": filename,
            "gradcam_image_data_url": None,
            "xai_error": None,
            "timings": {
                "prediction_seconds": round(prediction_duration_seconds, 4),
                "gradcam_seconds": None,
            },
        }

        if include_gradcam:
            try:
                gradcam_started_at = time.perf_counter()
                gradcam = build_gradcam_explanation(
                    image=image,
                    convnext_checkpoint_path=CONVNEXT_PATH,
                    target_idx=prediction["prediction"]["class_index"],
                )
                gradcam_duration_seconds = time.perf_counter() - gradcam_started_at
                INFERENCE_TOTAL.labels(stage="gradcam", status="success").inc()
                INFERENCE_DURATION_SECONDS.labels(stage="gradcam").observe(gradcam_duration_seconds)
                response_payload["timings"]["gradcam_seconds"] = round(gradcam_duration_seconds, 4)
                response_payload["gradcam_image_data_url"] = pil_to_data_url(gradcam["overlay"], fmt="PNG")
            except Exception as exc:
                INFERENCE_TOTAL.labels(stage="gradcam", status="error").inc()
                response_payload["xai_error"] = (
                    "Prediction completed, but Grad-CAM could not be generated. "
                    f"{exc}"
                )

        return jsonify(response_payload)
    except Exception as exc:
        INFERENCE_TOTAL.labels(stage="prediction", status="error").inc()
        return jsonify({"status": "error", "error": str(exc)}), 500


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default=os.getenv("MODEL_SERVICE_HOST", "0.0.0.0"))
    parser.add_argument("--port", type=int, default=int(os.getenv("MODEL_SERVICE_PORT", "8001")))
    args = parser.parse_args()
    app.run(host=args.host, port=args.port, debug=False)


if __name__ == "__main__":
    initialize_metrics()
    main()
