"""Evictors: ask an inference server to release memory, and restore it after.

An evictor implements three verbs: evict() asks the server to let go,
settled() answers whether it has, restore() brings a model back once the
training job finishes. Anything with those verbs plugs into EvictStage.
"""

from __future__ import annotations

import json
import time
import urllib.request


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

    def _get(self, path: str) -> dict:
        with urllib.request.urlopen(f"{self.base_url}{path}", timeout=self._timeout) as resp:
            return json.load(resp)

    def _post(self, path: str, payload: dict) -> None:
        req = urllib.request.Request(
            f"{self.base_url}{path}",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=self._timeout):
            pass

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
