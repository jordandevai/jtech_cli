"""File tools: /read, /write, /diff. Chunked reads to avoid blowing up the terminal."""

from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path

RANGE_RE = re.compile(r"^(.+?)(?::(\d+)(?:-(\d+))?)?$")


def _resolve(path: str) -> Path:
    p = Path(path).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def cmd_read(arg: str) -> str:
    m = RANGE_RE.match(arg.strip())
    if not m:
        return "Usage: /read PATH[:LINE]  e.g. main.py or main.py:10-40"
    path = _resolve(m.group(1))
    if not path.is_file():
        return f"No such file: {path}"
    lines = path.read_text().splitlines()
    total = len(lines)

    start = int(m.group(2)) if m.group(2) else 1
    end = int(m.group(3)) if m.group(3) else start if m.group(2) else total
    start, end = max(1, start), min(total, end)
    if start > end:
        return f"Range out of bounds (file has {total} lines)."

    width = len(str(end))
    out = [f"{path}  ({total} lines, showing {start}-{end}):", ""]
    out += [f"{i:>{width}} | {lines[i-1]}" for i in range(start, end + 1)]
    return "\n".join(out)


def cmd_write(path_arg: str, content: str) -> str:
    path = _resolve(path_arg.strip())
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content)
    n = content.count("\n") + 1
    return f"Wrote {n} lines to {path}"


def cmd_diff(path_arg: str) -> str:
    path = _resolve(path_arg.strip())
    if not path.is_file():
        return f"No such file: {path}"
    fd, tmp_name = tempfile.mkstemp(suffix=".tmp")
    os.close(fd)
    try:
        with open(tmp_name, "w") as fh:
            fh.write(path.read_text())
    except OSError as e:
        os.unlink(tmp_name)
        return f"Could not create temp file: {e}"
    return (
        f"Temp copy of {path} saved to {tmp_name}.\n"
        "Edit it, then diff with:\n"
        f"  diff {path} {tmp_name}\n"
        "Or use the /write command to apply changes."
    )
