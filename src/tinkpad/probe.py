"""Is-this-checkpoint-actually-serving probe.

We use the Tinker SDK's `SamplingClient.sample()` (NOT the OpenAI-compatible
HTTP endpoint). The OAI endpoint requires an active "session" that we don't
control, and returns 404 "Invalid session" outside of one — useless for a
probe. The SDK path works directly against any sampler checkpoint URI.

Caveats:
  - First call against a cold sampler can take ~10–30s while the checkpoint
    is loaded. Subsequent calls on the same path are near-instant.
  - Only sampler checkpoints (`.../sampler_weights/...`) can be probed.
    Training checkpoints are returned as `skipped`.
"""
from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Iterable

import tinker
from tinker import EncodedTextChunk, ModelInput, SamplingParams

from .config import api_key

PROBE_PROMPT = "Say the word OK."
PROBE_TIMEOUT_S = 60.0
DEFAULT_CONCURRENCY = 4


@dataclass(frozen=True)
class ProbeResult:
    tinker_path: str
    status: str  # ok | fail | timeout | skipped
    latency_ms: int | None
    sample: str | None
    error: str | None

    @property
    def emoji(self) -> str:
        return {"ok": "✓", "fail": "✗", "timeout": "⌛", "skipped": "-"}.get(self.status, "?")


def probe(tinker_path: str, *, key: str | None = None, timeout: float = PROBE_TIMEOUT_S) -> ProbeResult:
    """Synchronous one-shot probe. Returns ProbeResult (never raises)."""
    if "sampler_weights" not in tinker_path:
        return ProbeResult(tinker_path, "skipped", None, None, "not a sampler checkpoint")

    key = key or api_key()
    if not key:
        return ProbeResult(tinker_path, "fail", None, None, "TINKER_API_KEY not set")

    t0 = time.perf_counter()
    try:
        sc = tinker.ServiceClient(api_key=key)
        sclient = sc.create_sampling_client(model_path=tinker_path)
        tok = sclient.get_tokenizer()
        ids = tok.encode(PROBE_PROMPT)
        prompt = ModelInput(chunks=[EncodedTextChunk(tokens=ids)])
        fut = sclient.sample(
            prompt=prompt,
            num_samples=1,
            sampling_params=SamplingParams(max_tokens=8, temperature=0.0),
        )
        resp = fut.result(timeout=timeout)
        text = tok.decode(resp.sequences[0].tokens)
    except TimeoutError:
        return ProbeResult(tinker_path, "timeout", int((time.perf_counter() - t0) * 1000), None, "sample timed out")
    except tinker.NotFoundError as e:
        return ProbeResult(tinker_path, "fail", int((time.perf_counter() - t0) * 1000), None, f"not found: {e}")
    except Exception as e:
        return ProbeResult(tinker_path, "fail", int((time.perf_counter() - t0) * 1000), None, f"{type(e).__name__}: {e}")
    latency_ms = int((time.perf_counter() - t0) * 1000)
    return ProbeResult(tinker_path, "ok", latency_ms, text[:80], None)


def probe_many(paths: Iterable[str], *, concurrency: int = DEFAULT_CONCURRENCY, timeout: float = PROBE_TIMEOUT_S, key: str | None = None) -> list[ProbeResult]:
    paths = list(paths)
    key = key or api_key()
    if not key:
        return [ProbeResult(p, "fail", None, None, "TINKER_API_KEY not set") for p in paths]

    results: dict[str, ProbeResult] = {}
    with ThreadPoolExecutor(max_workers=max(1, concurrency)) as ex:
        futs = {ex.submit(probe, p, key=key, timeout=timeout): p for p in paths}
        for f in as_completed(futs):
            p = futs[f]
            try:
                results[p] = f.result()
            except Exception as e:
                results[p] = ProbeResult(p, "fail", None, None, f"{type(e).__name__}: {e}")
    # Preserve input order
    return [results[p] for p in paths]
