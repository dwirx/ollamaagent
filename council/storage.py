from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from .types import DebateState


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def autosave_json(state: DebateState, out_dir: Path = Path("debates")) -> None:
    ensure_dir(out_dir)
    ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    title = (state.config.title or "Debate").replace(" ", "_")[:48]
    filename = f"{ts}_{title}.json"
    tmp = out_dir / (filename + ".tmp")
    final = out_dir / filename
    data = state.model_dump(mode="json")
    tmp.write_text(json.dumps(data, ensure_ascii=False, indent=2))
    tmp.replace(final)


