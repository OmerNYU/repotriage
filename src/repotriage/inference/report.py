"""JSON serialization for inference responses."""

from __future__ import annotations

import json

from repotriage.inference.models import InferenceResponse


def format_inference_response_json(response: InferenceResponse, *, pretty: bool = False) -> str:
    """Serialize an inference response to JSON."""
    payload = response.model_dump(mode="json")
    if pretty:
        return json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=False) + "\n"
