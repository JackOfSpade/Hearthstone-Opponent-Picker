"""Silent debug log - Layer 6.

Standard: *"Silent debug log (per-run action journal + before/after screencaps
and sensor snippets on any anomaly) makes every halt reconstructable offline at
~no runtime cost."*

The journal is always on (cheap: append-only JSONL). Frames/sensor snippets are
persisted only on an anomaly, so a green run costs nothing but a few KB of text.
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field, is_dataclass
from pathlib import Path
from typing import Any

from .perception.image import Frame, pil_available


@dataclass
class JournalEntry:
    t: float
    kind: str
    detail: dict[str, Any] = field(default_factory=dict)


class DebugLog:
    def __init__(self, run_dir: str | Path, clock=time.time):
        self.run_dir = Path(run_dir)
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self._journal = self.run_dir / "journal.jsonl"
        self._clock = clock
        self._n = 0

    def record(self, kind: str, **detail: Any) -> None:
        entry = JournalEntry(t=self._clock(), kind=kind, detail=_jsonable(detail))
        with self._journal.open("a") as f:
            f.write(json.dumps(asdict(entry)) + "\n")

    def anomaly(
        self,
        reason: str,
        before: Frame | None = None,
        after: Frame | None = None,
        **context: Any,
    ) -> None:
        """Record an anomaly and, if possible, dump before/after frames."""
        self._n += 1
        self.record("anomaly", reason=reason, index=self._n, **context)
        if pil_available():
            for tag, frame in (("before", before), ("after", after)):
                if frame is not None:
                    self._save_frame(frame, f"anomaly_{self._n}_{tag}.png")

    def _save_frame(self, frame: Frame, name: str) -> None:  # pragma: no cover - needs Pillow
        try:
            from PIL import Image
            import numpy as np
            path = self.run_dir / name
            if isinstance(frame.data, (bytes, bytearray)):
                Image.frombytes("L", (frame.width, frame.height), bytes(frame.data)).save(path)
            else:
                Image.fromarray(np.asarray(frame.data, dtype="uint8"), "L").save(path)
        except Exception:
            pass


def _jsonable(d: dict[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for k, v in d.items():
        if is_dataclass(v):
            out[k] = asdict(v)
        elif isinstance(v, (str, int, float, bool)) or v is None:
            out[k] = v
        elif isinstance(v, (list, tuple)):
            out[k] = [x if isinstance(x, (str, int, float, bool)) or x is None else str(x) for x in v]
        else:
            out[k] = str(v)
    return out
