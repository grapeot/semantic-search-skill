from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def counter_enabled(explicit: bool | None = None) -> bool:
    if explicit is not None:
        return explicit
    return os.environ.get("SEMANTIC_SEARCH_COUNTER_ENABLED", "0").lower() in {"1", "true", "yes", "on"}


def counter_path(cache_dir: Path, explicit: str | None = None) -> Path:
    value = explicit or os.environ.get("SEMANTIC_SEARCH_COUNTER_PATH")
    if value:
        path = Path(value)
        return path if path.is_absolute() else Path.cwd() / path
    return cache_dir / "counter.jsonl"


def append_counter_event(path: Path, event: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {"timestamp": datetime.now(timezone.utc).isoformat(), **event}
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
