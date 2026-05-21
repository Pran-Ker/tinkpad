"""Public Python helpers — called from training scripts, notebooks, anywhere.

Kept small and dependency-light so importing `tinkpad` in a script never
pulls in textual, typer, etc.
"""
from __future__ import annotations

import os
from pathlib import Path

from .registry import Registry


def register_current_run(run_id: str, name: str | None = None) -> str:
    """Map a Tinker run_id to a human-friendly experiment name.

    Call this from your training script right after creating the run, e.g.:

        run = service.create_lora_training_client(...)
        tinkpad.register_current_run(run.run_id, "long-context-test")

    If `name` is omitted, the current working directory's folder name is used.
    Returns the name that was actually stored.
    """
    if name is None:
        name = Path(os.getcwd()).name
    reg = Registry()
    reg.set(run_id, name, note="from register_current_run")
    reg.save()
    return name
