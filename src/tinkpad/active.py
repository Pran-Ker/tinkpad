"""Active checkpoint pointer.

`tinkpad use <path>` writes the URI to ~/.tinkpad/active in two forms:

  1. The bare URI on the first line (so `cat ~/.tinkpad/active` is useful).
  2. A sourceable shell snippet at ~/.tinkpad/active.env exporting
     TINKPAD_CKPT (and OPENAI_BASE_URL/OPENAI_API_KEY for convenience).

Downstream tools can read either form.
"""
from __future__ import annotations

import shlex
from pathlib import Path

from ._atomic import atomic_write_text
from .config import ACTIVE_PATH, TINKPAD_DIR, TINKER_OAI_BASE_URL, ensure_dir


def set_active(tinker_path: str) -> Path:
    ensure_dir()
    # Write the env file FIRST so a sourcing reader never sees a new pointer
    # paired with a stale env file.
    env = TINKPAD_DIR / "active.env"
    atomic_write_text(
        env,
        f"export TINKPAD_CKPT={shlex.quote(tinker_path)}\n"
        f"export OPENAI_BASE_URL={shlex.quote(TINKER_OAI_BASE_URL)}\n"
        'export OPENAI_API_KEY="${TINKER_API_KEY:-$OPENAI_API_KEY}"\n',
    )
    atomic_write_text(ACTIVE_PATH, tinker_path + "\n")
    return ACTIVE_PATH


def get_active() -> str | None:
    if not ACTIVE_PATH.exists():
        return None
    text = ACTIVE_PATH.read_text().strip()
    return text or None


def clear_active() -> None:
    if ACTIVE_PATH.exists():
        ACTIVE_PATH.unlink()
    env = TINKPAD_DIR / "active.env"
    if env.exists():
        env.unlink()
