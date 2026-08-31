"""Run the training job at reduced priority, and restore the server after.

The child runs nicer than the inference server so a busy trainer degrades
serving latency instead of starving it. Restoration runs whether the child
succeeded or not - the box goes back to serving either way.
"""

from __future__ import annotations

import os
import subprocess
import sys


def launch_low_priority(cmd: list[str], _popen=subprocess.Popen) -> int:
    """Run `cmd` as a reduced-priority child; return its exit code."""
    kwargs: dict = {}
    if sys.platform == "win32":
        kwargs["creationflags"] = subprocess.BELOW_NORMAL_PRIORITY_CLASS
    else:
        kwargs["preexec_fn"] = lambda: os.nice(10)
    child = _popen(cmd, **kwargs)
    return child.wait()


def run_then_restore(cmd: list[str], evictor=None, restore_model: str | None = None) -> int:
    """Launch the job, then hand the box back to the inference server."""
    try:
        return launch_low_priority(cmd)
    finally:
        if evictor is not None:
            try:
                evictor.restore(restore_model)
            except Exception:
                pass  # restoration is best-effort; the exit code belongs to the job
