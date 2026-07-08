"""A minimal, comment-preserving TOML writer.

`hop calibrate` and the control app both need to change one key in the user's
config without disturbing the rest of it. A parse/re-emit round-trip through
``tomllib`` + a serializer would discard every comment - and this config is mostly
comments, because the standard requires a provenance note beside any constant you
change ("undocumented constants rot").

So this is deliberately line-based rather than a real TOML implementation. It
handles exactly what we write: scalar and inline-array values under top-level
tables. Anything more exotic belongs in a real library.
"""

from __future__ import annotations

import re

_TABLE_RE = re.compile(r"^\s*\[([^\[\]]+)\]\s*(?:#.*)?$")


def _key_re(key: str) -> re.Pattern[str]:
    return re.compile(rf"^\s*{re.escape(key)}\s*=")


def upsert_toml_scalar(text: str, table: str, key: str, value: str,
                       comment: str | None = None) -> str:
    """Set ``key = value`` under ``[table]``, creating or replacing in place.

    ``hop calibrate`` used to *append* its results, which emits a second
    ``[motor]`` header on the second run - and duplicate tables are invalid TOML,
    so re-calibrating would leave a config that no ``hop`` command could load.
    This keeps the write idempotent.

    Deliberately line-based rather than a full TOML round-trip: it preserves the
    user's comments and provenance notes, which a parse/re-emit would discard.
    """
    line = f"{key} = {value}" + (f"    # {comment}" if comment else "")
    lines = text.splitlines()

    start = None
    for i, raw in enumerate(lines):
        m = _TABLE_RE.match(raw)
        if m and m.group(1).strip() == table:
            start = i
            break

    if start is None:
        prefix = lines + ([""] if lines and lines[-1].strip() else [])
        return "\n".join([*prefix, f"[{table}]", line, ""])

    end = len(lines)
    for j in range(start + 1, len(lines)):
        if _TABLE_RE.match(lines[j]):
            end = j
            break

    pat = _key_re(key)
    for j in range(start + 1, end):
        if pat.match(lines[j]):
            lines[j] = line
            return "\n".join(lines) + "\n"

    insert = end
    while insert > start + 1 and not lines[insert - 1].strip():
        insert -= 1
    lines.insert(insert, line)
    return "\n".join(lines) + "\n"
