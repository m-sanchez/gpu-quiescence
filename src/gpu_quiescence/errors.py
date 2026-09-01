"""One error type that must never be confused with a verdict.

A missing prerequisite, an unreachable server, or a command that does not
exist is a *usage* problem: the handshake could not run, so it has no
opinion on whether the box is ready. Exit code 2, never exit code 1.
"""

from __future__ import annotations


class UsageError(RuntimeError):
    """A missing prerequisite is a usage problem, not a 'not ready' verdict."""
