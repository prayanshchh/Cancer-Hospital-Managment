import base64
import json
import os
from typing import Any, Dict
from urllib import error, request


DEFAULT_MODEL_API_URL = "http://127.0.0.1:8001"
DEFAULT_TIMEOUT_SECONDS = 180.0


class ModelServiceError(RuntimeError):
    pass


def get_model_api_url() -> str:
    return os.getenv("MODEL_API_URL", DEFAULT_MODEL_API_URL).rstrip("/")


def get_model_api_timeout() -> float:
    raw_value = os.getenv("MODEL_API_TIMEOUT_SECONDS", str(DEFAULT_TIMEOUT_SECONDS)).strip()
    try:
        return float(raw_value)
    except ValueError:
        return DEFAULT_TIMEOUT_SECONDS


def _decode_json_response(response: Any) -> Dict[str, Any]:
    charset = "utf-8"
    if hasattr(response, "headers"):
        charset = response.headers.get_content_charset("utf-8")
    payload = response.read().decode(charset)
    return json.loads(payload) if payload else {}


def _extract_error_message(exc: error.HTTPError) -> str:
    try:
        payload = _decode_json_response(exc)
    except Exception:
        payload = {}
    if isinstance(payload, dict) and payload.get("error"):
        return str(payload["error"])
    return exc.reason or f"HTTP {exc.code}"


def fetch_model_service_health(timeout_seconds: float = 2.5) -> Dict[str, Any]:
    endpoint = f"{get_model_api_url()}/healthz"
    req = request.Request(endpoint, method="GET")
    try:
        with request.urlopen(req, timeout=timeout_seconds) as response:
            return _decode_json_response(response)
    except error.HTTPError as exc:
        raise ModelServiceError(_extract_error_message(exc)) from exc
    except error.URLError as exc:
        raise ModelServiceError(f"Could not reach model service at {endpoint}: {exc.reason}") from exc


def predict_with_model_service(
    image_bytes: bytes,
    filename: str = "upload.png",
    include_gradcam: bool = True,
) -> Dict[str, Any]:
    endpoint = f"{get_model_api_url()}/predict"
    payload = {
        "image_base64": base64.b64encode(image_bytes).decode("utf-8"),
        "filename": filename,
        "include_gradcam": include_gradcam,
    }
    encoded_payload = json.dumps(payload).encode("utf-8")
    req = request.Request(
        endpoint,
        data=encoded_payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with request.urlopen(req, timeout=get_model_api_timeout()) as response:
            payload = _decode_json_response(response)
            if payload.get("status") != "ok" or "prediction" not in payload:
                raise ModelServiceError(str(payload.get("error") or "Model service returned an invalid response."))
            return payload
    except error.HTTPError as exc:
        raise ModelServiceError(_extract_error_message(exc)) from exc
    except error.URLError as exc:
        raise ModelServiceError(f"Could not reach model service at {endpoint}: {exc.reason}") from exc
