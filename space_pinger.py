from typing import List, Tuple
from gradio_client import Client

def _is_live(space_id: str) -> Tuple[bool, str]:
    try:
        c = Client(space_id, hf_token=None)
        c.view_api(all_endpoints=True)
        return True, "OK"
    except Exception as e:
        return False, f"{e.__class__.__name__}: {e}"

def prioritize_live_spaces(space_ids: List[str]) -> Tuple[List[str], List[str]]:
    logs: List[str] = []
    live, dead = [], []
    for sid in space_ids:
        ok, msg = _is_live(sid)
        logs.append(f"{sid} -> {msg}")
        (live if ok else dead).append(sid)
    return live + dead, logs
