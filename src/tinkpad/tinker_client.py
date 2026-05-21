"""Thin adapter around tinker SDK's RestClient.

Wraps the futures-based SDK calls so the rest of tinkpad never has to deal
with `.result()` and gives us a small, testable surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Iterable

import tinker

from .config import api_key


@dataclass(frozen=True)
class Checkpoint:
    checkpoint_id: str
    type: str  # "training" | "sampler"
    tinker_path: str
    size_bytes: int
    created_at: datetime | None
    public: bool
    expires_at: datetime | None
    run_id: str

    @property
    def short_path(self) -> str:
        # tinker://<uuid>:train:0/sampler_weights/000030 -> sampler_weights/000030
        return self.tinker_path.split("/", 3)[-1]


@dataclass(frozen=True)
class Run:
    run_id: str
    base_model: str
    is_lora: bool
    lora_rank: int | None
    corrupted: bool
    last_request_time: datetime | None
    last_checkpoint_path: str | None
    last_sampler_checkpoint_path: str | None


class TinkerClient:
    def __init__(self, key: str | None = None):
        key = key or api_key()
        if not key:
            raise RuntimeError(
                "TINKER_API_KEY not set. Source ~/.local/secrets or export it."
            )
        self._sc = tinker.ServiceClient(api_key=key)
        self._rc = self._sc.create_rest_client()

    # ---------- runs ----------
    def list_runs(self) -> list[Run]:
        resp = self._rc.list_training_runs().result()
        return [_to_run(r) for r in resp.training_runs]

    def get_run(self, run_id: str) -> Run:
        run_id = _strip_run_id(run_id)
        r = self._rc.get_training_run(run_id).result()
        return _to_run(r)

    # ---------- checkpoints ----------
    def list_checkpoints(self, run_id: str | None = None) -> list[Checkpoint]:
        if run_id:
            run_id = _strip_run_id(run_id)
            resp = self._rc.list_checkpoints(run_id).result()
        else:
            resp = self._rc.list_user_checkpoints(limit=500).result()
        return [_to_ckpt(c) for c in resp.checkpoints]

    def set_ttl(self, tinker_path: str, ttl_seconds: int | None) -> None:
        if ttl_seconds is None:
            self._rc.set_checkpoint_ttl_from_tinker_path(tinker_path, remove=True).result()
        else:
            self._rc.set_checkpoint_ttl_from_tinker_path(tinker_path, ttl_seconds=ttl_seconds).result()

    def delete(self, tinker_path: str) -> None:
        self._rc.delete_checkpoint_from_tinker_path(tinker_path).result()

    def publish(self, tinker_path: str) -> None:
        self._rc.publish_checkpoint_from_tinker_path(tinker_path).result()

    def unpublish(self, tinker_path: str) -> None:
        self._rc.unpublish_checkpoint_from_tinker_path(tinker_path).result()


def _strip_run_id(run_id: str) -> str:
    rid = run_id.removeprefix("tinker://")
    rid = rid.split("/", 1)[0]
    return rid


def _to_run(r) -> Run:
    last_ck = r.last_checkpoint.tinker_path if r.last_checkpoint else None
    last_sk = r.last_sampler_checkpoint.tinker_path if r.last_sampler_checkpoint else None
    return Run(
        run_id=r.training_run_id,
        base_model=r.base_model,
        is_lora=bool(r.is_lora),
        lora_rank=r.lora_rank,
        corrupted=bool(r.corrupted),
        last_request_time=r.last_request_time,
        last_checkpoint_path=last_ck,
        last_sampler_checkpoint_path=last_sk,
    )


def _to_ckpt(c) -> Checkpoint:
    # tinker://<run_id>/sampler_weights/...
    parsed = c.tinker_path.removeprefix("tinker://").split("/", 1)
    run_id = parsed[0]
    return Checkpoint(
        checkpoint_id=c.checkpoint_id,
        type=str(c.checkpoint_type),
        tinker_path=c.tinker_path,
        size_bytes=c.size_bytes or 0,
        created_at=c.time,
        public=bool(c.public),
        expires_at=c.expires_at,
        run_id=run_id,
    )
