import json
import os
import uuid

import httpx

BASE_URL = os.environ.get("BACKEND_URL", "http://localhost:8000")

# Regular calls (config, models, upload) are quick; the /run stream can
# run long across retries, so its read timeout is unbounded while connect
# stays short — a dead backend should fail fast, a slow *agent* shouldn't.
DEFAULT_TIMEOUT = httpx.Timeout(10.0, read=30.0)
STREAM_TIMEOUT = httpx.Timeout(10.0, read=None)


class BackendUnavailableError(RuntimeError):
    """Raised when the backend can't be reached at all — distinct from a
    request that reached the backend and got a real error response, so
    the caller can tell 'not running' apart from 'ran and failed'."""


def _raise_if_unreachable(exc: httpx.TransportError):
    raise BackendUnavailableError(
        f"Could not reach the backend at {BASE_URL} — "
        f"is `uvicorn backend.main:app` running? ({exc})"
    ) from exc


def get_config() -> dict:
    """GET /config -> {"max_retries": int, "quality_threshold": float,
    "has_key": bool, "masked_key": str}"""
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.get(f"{BASE_URL}/config")
    except httpx.TransportError as exc:
        _raise_if_unreachable(exc)
    resp.raise_for_status()
    return resp.json()


def get_api_key_status() -> dict:
    """Same /config endpoint as get_config() — the key status and the
    tuning constants are cheap to fetch together, so the backend serves
    both from one route rather than two."""
    config = get_config()
    return {"has_key": config["has_key"], "masked": config["masked_key"]}


def set_api_key(new_key: str) -> dict:
    """POST /config {"api_key": new_key} -> {"status": "ok"}"""
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.post(f"{BASE_URL}/config", json={"api_key": new_key})
    except httpx.TransportError as exc:
        _raise_if_unreachable(exc)
    resp.raise_for_status()
    return resp.json()


def get_available_models() -> list[str]:
    """GET /models -> {"models": list[str]}"""
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.get(f"{BASE_URL}/models")
    except httpx.TransportError as exc:
        _raise_if_unreachable(exc)
    resp.raise_for_status()
    return resp.json()["models"]


def upload_files(session_id: str, files: list) -> dict:
    """POST /upload?session_id=... (multipart) -> {"status", "files_indexed"}

    `files` is the list st.file_uploader returns — UploadedFile objects
    with .name and .getvalue().
    """
    multipart = [
        ("files", (f.name, f.getvalue(), f.type or "application/octet-stream"))
        for f in files
    ]
    try:
        with httpx.Client(timeout=DEFAULT_TIMEOUT) as client:
            resp = client.post(
                f"{BASE_URL}/upload",
                params={"session_id": session_id},
                files=multipart,
            )
    except httpx.TransportError as exc:
        _raise_if_unreachable(exc)
    resp.raise_for_status()
    return resp.json()


def _iter_sse_events(response: httpx.Response):
    """Parse `data: <json>` lines from an SSE response into dicts. Blank
    lines separate events and are skipped; anything not prefixed with
    'data:' (e.g. a future 'event:' line) is ignored for now."""
    for line in response.iter_lines():
        if not line or not line.startswith("data:"):
            continue
        payload = line[len("data:") :].strip()
        if payload:
            yield json.loads(payload)


def run_agent_stream(session_id: str, goal: str, model: str, temperature: float):
    """POST /run {"session_id", "goal", "model", "temperature"} -> SSE stream
    of {"node": str, "output": dict} events, one per node firing, in the
    same shape the mock version produced.
    """
    payload = {
        "session_id": session_id,
        "goal": goal,
        "model": model,
        "temperature": temperature,
    }
    try:
        with httpx.Client(timeout=STREAM_TIMEOUT) as client:
            with client.stream("POST", f"{BASE_URL}/run", json=payload) as response:
                response.raise_for_status()
                yield from _iter_sse_events(response)
    except httpx.TransportError as exc:
        _raise_if_unreachable(exc)


def new_session_id() -> str:
    return str(uuid.uuid4())
