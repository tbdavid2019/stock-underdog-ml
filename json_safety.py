"""Helpers for validating values before sending them to JSON APIs."""

import json
import math
from typing import Any


def sanitize_json_value(value: Any) -> Any:
    """Convert non-finite numeric values to JSON null recursively."""
    if isinstance(value, dict):
        return {key: sanitize_json_value(item) for key, item in value.items()}

    if isinstance(value, (list, tuple)):
        return [sanitize_json_value(item) for item in value]

    if isinstance(value, float):
        return value if math.isfinite(value) else None

    # Convert numpy scalar values without importing numpy into this utility.
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return sanitize_json_value(item())
        except (TypeError, ValueError):
            pass

    return value


def validate_json_payload(value: Any) -> None:
    """Raise ValueError when a value cannot be encoded as strict JSON."""
    try:
        json.dumps(value, allow_nan=False)
    except (TypeError, ValueError) as exc:
        raise ValueError("payload is not valid strict JSON") from exc
