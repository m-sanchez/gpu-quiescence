"""Evictors: ask an inference server to release memory, and restore it after.

An evictor implements three verbs: evict() asks the server to let go,
settled() answers whether it has, restore() brings a model back once the
training job finishes. Anything with those verbs plugs into EvictStage.
"""

from __future__ import annotations

import json
import time
import urllib.error
import urllib.request

from .errors import UsageError


class OllamaEvictor:
    """Evict and re-warm models on an Ollama server via its public API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 20.0,
        poll_interval_s: float = 1.0,
        _sleep=time.sleep,
        _clock=time.monotonic,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout_s
        self._poll = poll_interval_s
        self._sleep = _sleep
        self._clock = _clock

    def _open(self, req) -> bytes:
        """Every failure to reach the server is a usage error, not a verdict.

        A typo in --ollama must not come back as "the box is not ready": the
        handshake never ran, so it has nothing to report.
        """
        url = req if isinstance(req, str) else req.full_url
        try:
            with urllib.request.urlopen(req, timeout=self._timeout) as resp:
                return resp.read()
        except urllib.error.HTTPError as exc:
            raise UsageError(f"{url} answered HTTP {exc.code} {exc.reason}") from exc
        except OSError as exc:  # URLError, timeouts, DNS, connection refused
            raise UsageError(f"cannot reach the inference server at {url}: {exc}") from exc

    def _get(self, path: str) -> dict:
        url = f"{self.base_url}{path}"
        raw = self._open(url)
        try:
            return json.loads(raw)
        except ValueError as exc:  # includes json.JSONDecodeError
            raise UsageError(f"{url} did not answer with JSON; is this an Ollama server?") from exc

    def _post(self, path: str, payload: dict) -> None:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        self._open(req)

    def loaded_models(self) -> list[str]:
        return [m.get("name", "") for m in self._get("/api/ps").get("models", [])]

    def evict(self) -> None:
        """Ask Ollama to drop every loaded model (keep_alive: 0)."""
        for name in self.loaded_models():
            self._post("/api/generate", {"model": name, "keep_alive": 0, "prompt": ""})

    def settled(self) -> bool:
        """Poll until the server reports nothing loaded, within the timeout."""
        deadline = self._clock() + self._timeout
        while True:
            if not self.loaded_models():
                return True
            if self._clock() >= deadline:
                return False
            self._sleep(self._poll)

    def restore(self, model: str | None = None) -> None:
        """Re-warm one model so the first real request does not pay the load."""
        if model:
            self._post("/api/generate", {"model": model, "prompt": ""})
