"""Shared Rich rendering helpers."""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Iterable

from rich.console import Console
from rich.table import Table
from rich import box

from .tinker_client import Checkpoint, Run

# Disable emoji substitution globally — Tinker run IDs contain ":train:"
# which Rich would otherwise render as 🚋.
console = Console(emoji=False)


def human_size(n: int) -> str:
    if n is None:
        return "-"
    units = ["B", "KB", "MB", "GB", "TB"]
    f = float(n)
    for u in units:
        if f < 1024 or u == units[-1]:
            return f"{f:.1f}{u}" if u != "B" else f"{int(f)}{u}"
        f /= 1024
    return f"{f:.1f}TB"


def human_age(dt: datetime | None) -> str:
    if dt is None:
        return "-"
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    delta = datetime.now(timezone.utc) - dt
    s = int(delta.total_seconds())
    if s < 60:
        return f"{s}s ago"
    if s < 3600:
        return f"{s // 60}m ago"
    if s < 86400:
        return f"{s // 3600}h ago"
    if s < 86400 * 30:
        return f"{s // 86400}d ago"
    return dt.strftime("%Y-%m-%d")


def short_run(run_id: str) -> str:
    return run_id.split("-", 1)[0]


def runs_table(runs: Iterable[Run], name_for) -> Table:
    t = Table(title="Tinker runs", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    t.add_column("experiment", style="bold")
    t.add_column("run", style="dim")
    t.add_column("model", no_wrap=True, overflow="fold")
    t.add_column("lora")
    t.add_column("last activity")
    t.add_column("last sampler")
    t.add_column("ckpt created")
    for r in runs:
        name = name_for(r.run_id) or "[dim]—[/dim]"
        lora = f"r={r.lora_rank}" if r.is_lora else "no"
        last_sk = r.last_sampler_checkpoint_path.split("/")[-1] if r.last_sampler_checkpoint_path else "—"
        # Prefer sampler ckpt time, fall back to training ckpt time.
        ckpt_t = r.last_sampler_checkpoint_created_at or r.last_checkpoint_created_at
        ckpt_age = human_age(ckpt_t) if ckpt_t else "—"
        row_style = "red" if r.corrupted else ""
        t.add_row(name, short_run(r.run_id), r.base_model, lora, human_age(r.last_request_time), last_sk, ckpt_age, style=row_style)
    return t


def checkpoints_table(ckpts: Iterable[Checkpoint], name_for, probe_status: dict[str, str] | None = None) -> Table:
    t = Table(title="Checkpoints", box=box.SIMPLE_HEAVY, header_style="bold cyan")
    t.add_column("experiment", style="bold")
    t.add_column("run", style="dim")
    t.add_column("checkpoint", no_wrap=True, overflow="fold")
    t.add_column("type")
    t.add_column("size")
    t.add_column("age")
    t.add_column("pub")
    if probe_status is not None:
        t.add_column("probe")
    for c in ckpts:
        name = name_for(c.run_id) or "[dim]—[/dim]"
        t_color = "magenta" if c.type == "sampler" else "yellow"
        row = [
            name,
            short_run(c.run_id),
            c.checkpoint_id,
            f"[{t_color}]{c.type}[/]",
            human_size(c.size_bytes),
            human_age(c.created_at),
            "●" if c.public else " ",
        ]
        if probe_status is not None:
            row.append(probe_status.get(c.tinker_path, "[dim]—[/dim]"))
        t.add_row(*row)
    return t
