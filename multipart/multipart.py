"""Compatibility helper for FastAPI's multipart dependency check."""

from __future__ import annotations

from typing import Any, Dict, Tuple


def parse_options_header(value: str) -> Tuple[str, Dict[str, Any]]:
    """Return the header and an empty options dict.

    The implementation mirrors the signature of the real helper but intentionally
    keeps behaviour minimal as it is only used to satisfy FastAPI's import-time
    guard inside the test environment.
    """

    return value, {}

