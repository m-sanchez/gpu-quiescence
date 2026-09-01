"""Evictors: ask an inference server to release memory, and restore it after.

An evictor implements three verbs: evict() asks the server to let go,
settled() answers whether it has *right now*, restore() brings a model back
once the training job finishes. Anything with those verbs plugs into
EvictStage.

settled() is a predicate, not a wait: the stage owns the deadline. Two
components polling against two deadlines is how a report comes to state a
time nobody waited.
"""

from __future__ import annotations

import json
import urllib.error
import urllib.request
from dataclasses import dataclass

from .errors import UsageError


@dataclass(frozen=True)
class LoadedModel:
    """One model the server is holding, and the VRAM it says it occupies."""

    name: str
    size_vram_mib: float


class OllamaEvictor:
    """Evict and re-warm models on an Ollama server via its public API."""

    def __init__(
        self,
        base_url: str = "http://127.0.0.1:11434",
        timeout_s: float = 20.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self._timeout = timeout_s  # per-request HTTP timeout, not a poll budget

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

    def loaded_models(self) -> list[LoadedModel]:
        """What the server says it holds - with the bytes, not just the names.

        The byte count is the whole point: it is what turns "the server says
        it unloaded" into something the stage can check against the driver.
        """
        models = []
        for m in self._get("/api/ps").get("models", []):
            try:
                vram = float(m.get("size_vram") or 0) / (1024 * 1024)
            except (TypeError, ValueError):
                vram = 0.0
            models.append(LoadedModel(name=m.get("name", ""), size_vram_mib=vram))
        return models

    def held_vram_mib(self) -> float:
        """Total VRAM the server currently claims to be holding, in MiB."""
        return sum(m.size_vram_mib for m in self.loaded_models())

    def evict(self) -> None:
        """Ask Ollama to drop every loaded model (keep_alive: 0)."""
        for model in self.loaded_models():
            self._post("/api/generate", {"model": model.name, "keep_alive": 0, "prompt": ""})

    def settled(self) -> bool:
        """Does the server report nothing loaded right now? One question, one answer."""
        return not self.loaded_models()

    def restore(self, model: str | None = None) -> None:
        """Re-warm one model so the first real request does not pay the load."""
        if model:
            self._post("/api/generate", {"model": model, "prompt": ""})
