from __future__ import annotations
from typing import List, Tuple
import requests, json

API = "https://huggingface.co/api/spaces/{space_id}"

def ping_space(space_id: str, timeout: float = 7.0) -> str:
    url = API.format(space_id=space_id)
    try:
        r = requests.get(url, timeout=timeout)
        if r.status_code == 404:
            return "404 Not Found"
        if r.status_code in (401, 403):
            return f"{r.status_code} Unauthorized/Gated"
        if r.status_code != 200:
            return f"HTTP {r.status_code}"
        try:
            data = r.json()
        except json.JSONDecodeError as e:
            return f"JSONDecodeError: {e}"
        runtime = (data.get("runtime") or {}).get("stage") or data.get("runtime", "")
        state = str(runtime) or "UNKNOWN"
        return f"STATE {state}"
    except requests.exceptions.RequestException as e:
        return f"RequestsError: {e}"

def prioritize_live_spaces(space_ids: List[str]) -> tuple[list[str], list[str]]:
    LIVE_MARKERS = ("STATE RUNNING","STATE 'RUNNING'","STATE running")
    results = []
    logs = []
    for sid in space_ids:
        status = ping_space(sid)
        logs.append(f"{sid} -> {status}")
        results.append((sid, status))
    live = [sid for sid, st in results if any(m in st for m in LIVE_MARKERS)]
    warm = [sid for sid, st in results if ("STATE BUILDING" in st or "UNKNOWN" in st or "JSONDecodeError" in st)]
    rest = [sid for sid, st in results if sid not in live and sid not in warm]
    return live + warm + rest, logs
