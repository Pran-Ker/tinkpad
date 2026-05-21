"""tinkpad CLI."""
from __future__ import annotations

import json
import os
import sys
import warnings
from pathlib import Path
from typing import Optional

# Silence a couple of harmless upstream warnings before importing tinker.
warnings.filterwarnings("ignore", category=DeprecationWarning, module="pydantic")
os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")

import typer
from rich.panel import Panel
from rich.text import Text

from . import __version__
from . import active as active_mod
from .formatting import (
    checkpoints_table,
    console,
    human_age,
    human_size,
    runs_table,
    short_run,
)
from .probe import probe as probe_one, probe_many
from .registry import Registry, scan, load_scan_roots, save_scan_roots
from .tinker_client import TinkerClient

app = typer.Typer(
    help="tinkpad — browse / probe / switch Tinker checkpoints.",
    no_args_is_help=True,
    add_completion=False,
    context_settings={"help_option_names": ["-h", "--help"]},
)


def _client() -> TinkerClient:
    try:
        return TinkerClient()
    except RuntimeError as e:
        console.print(f"[red]{e}[/red]")
        raise typer.Exit(2)


def _name_for(reg: Registry):
    return lambda rid: reg.name_for(rid)


def _resolve_run_id(client: TinkerClient, query: str) -> str:
    """Resolve a short prefix or experiment name to a full run_id.

    Raises typer.Exit if 0 or >1 matches.
    """
    reg = Registry()
    runs = client.list_runs()
    # Strip tinker:// prefix in case the user pasted a full path
    q = query.removeprefix("tinker://").split("/", 1)[0]
    matches = []
    for r in runs:
        name = reg.name_for(r.run_id) or ""
        if r.run_id == q or r.run_id.startswith(q) or (name and q in name):
            matches.append(r.run_id)
    if len(matches) == 1:
        return matches[0]
    if not matches:
        console.print(f"[red]no run matches '{query}'[/]")
        raise typer.Exit(1)
    console.print(f"[red]ambiguous '{query}' — matches:[/]")
    for m in matches[:10]:
        console.print(f"  {m}")
    raise typer.Exit(1)


# ---------- top-level ----------
@app.callback(invoke_without_command=True)
def _root(
    ctx: typer.Context,
    version: bool = typer.Option(False, "--version", "-V"),
):
    if version:
        console.print(f"tinkpad {__version__}")
        raise typer.Exit()


# ---------- ls / runs / ckpts ----------
@app.command("ls")
def ls(
    run: Optional[str] = typer.Option(None, "--run", "-r", help="Filter by run_id substring or experiment name."),
    sampler_only: bool = typer.Option(False, "--sampler", "-s"),
    training_only: bool = typer.Option(False, "--training", "-t"),
    probe_all: bool = typer.Option(False, "--probe", "-p", help="Probe every sampler checkpoint (slow)."),
    limit: int = typer.Option(200, "--limit", "-n"),
):
    """List checkpoints across all runs, joined with local experiment names."""
    reg = Registry()
    scan(load_scan_roots())  # cheap-ish refresh
    reg = Registry()  # reload after scan
    client = _client()

    runs = {r.run_id.split(":", 1)[0]: r for r in client.list_runs()}
    if run:
        # resolve as substring against run_id OR experiment name
        keep = set()
        for rid in runs:
            full = runs[rid].run_id
            name = reg.name_for(full) or ""
            if run in full or (name and run in name):
                keep.add(rid)
        runs = {k: v for k, v in runs.items() if k in keep}

    all_ckpts: list = []
    if run and len(runs) == 1:
        rid = next(iter(runs.values())).run_id
        all_ckpts = client.list_checkpoints(rid)
    else:
        all_ckpts = client.list_checkpoints()
        if run:
            keep_ids = set(runs.keys())
            all_ckpts = [c for c in all_ckpts if c.run_id.split(":", 1)[0] in keep_ids]

    if sampler_only:
        all_ckpts = [c for c in all_ckpts if c.type == "sampler"]
    if training_only:
        all_ckpts = [c for c in all_ckpts if c.type == "training"]
    all_ckpts = all_ckpts[:limit]

    probe_status = None
    if probe_all:
        sampler_paths = [c.tinker_path for c in all_ckpts if c.type == "sampler"]
        results = probe_many(sampler_paths)
        probe_status = {}
        for r in results:
            color = {"ok": "green", "fail": "red", "timeout": "yellow", "skipped": "dim"}[r.status]
            lat = f" {r.latency_ms}ms" if r.latency_ms is not None else ""
            probe_status[r.tinker_path] = f"[{color}]{r.emoji} {r.status}{lat}[/]"

    console.print(checkpoints_table(all_ckpts, _name_for(reg), probe_status))
    active = active_mod.get_active()
    if active:
        console.print(f"[dim]active:[/] [bold]{active}[/]")


@app.command("runs")
def runs():
    """List training runs with last-activity and last-sampler info."""
    reg = Registry()
    scan(load_scan_roots())
    reg = Registry()
    client = _client()
    rs = client.list_runs()
    console.print(runs_table(rs, _name_for(reg)))


# ---------- info ----------
@app.command("info")
def info(path: str = typer.Argument(..., help="tinker:// checkpoint path or 'active'.")):
    """Detail view of one checkpoint (with probe)."""
    reg = Registry()
    client = _client()
    if path == "active":
        a = active_mod.get_active()
        if not a:
            console.print("[red]no active checkpoint set[/]")
            raise typer.Exit(1)
        path = a
    # Find checkpoint by listing the run
    rid = path.removeprefix("tinker://").split("/", 1)[0]
    try:
        ckpts = client.list_checkpoints(rid)
    except Exception as e:
        console.print(f"[red]could not fetch run {rid}: {e}[/]")
        raise typer.Exit(1)
    match = next((c for c in ckpts if c.tinker_path == path), None)
    if not match:
        console.print(f"[red]no such checkpoint: {path}[/]")
        raise typer.Exit(1)
    name = reg.name_for(rid) or "(unregistered)"
    from rich.markup import escape as _esc
    body = [
        f"[bold]experiment[/] {_esc(name)}",
        f"[bold]run[/]        {short_run(rid)}  ([dim]{_esc(rid)}[/])",
        f"[bold]path[/]       {_esc(match.tinker_path)}",
        f"[bold]type[/]       {match.type}",
        f"[bold]size[/]       {human_size(match.size_bytes)}",
        f"[bold]created[/]    {human_age(match.created_at)}  ({_esc(str(match.created_at))})",
        f"[bold]public[/]     {match.public}",
        f"[bold]expires[/]    {_esc(str(match.expires_at)) if match.expires_at else '—'}",
    ]
    console.print(Panel("\n".join(body), title=match.checkpoint_id, border_style="cyan"))

    if match.type == "sampler":
        console.print("[dim]probing…[/]")
        res = probe_one(match.tinker_path)
        color = {"ok": "green", "fail": "red", "timeout": "yellow", "skipped": "dim"}[res.status]
        lat = f" {res.latency_ms}ms" if res.latency_ms else ""
        sample = f"\n[dim]sample:[/] {res.sample!r}" if res.sample else ""
        err = f"\n[red]error:[/] {res.error}" if res.error else ""
        console.print(Panel(f"[{color}]{res.emoji} {res.status}{lat}[/]{sample}{err}", title="probe", border_style=color))


# ---------- probe ----------
@app.command("probe")
def probe_cmd(
    paths: list[str] = typer.Argument(None, help="One or more tinker:// paths; default = active."),
    run: Optional[str] = typer.Option(None, "--run", "-r", help="Probe every sampler ckpt in a run."),
    all_samplers: bool = typer.Option(False, "--all", "-a", help="Probe every sampler across all runs (slow)."),
):
    """Fire a tiny inference to verify a checkpoint actually serves."""
    client = _client()
    target_paths: list[str] = []
    if all_samplers:
        target_paths = [c.tinker_path for c in client.list_checkpoints() if c.type == "sampler"]
    elif run:
        full_rid = _resolve_run_id(client, run)
        target_paths = [c.tinker_path for c in client.list_checkpoints(full_rid) if c.type == "sampler"]
    elif paths:
        target_paths = list(paths)
    else:
        a = active_mod.get_active()
        if not a:
            console.print("[red]no path given and no active checkpoint[/]")
            raise typer.Exit(2)
        target_paths = [a]

    if not target_paths:
        console.print("[yellow]no sampler checkpoints to probe[/]")
        return
    results = probe_many(target_paths)
    for r in results:
        color = {"ok": "green", "fail": "red", "timeout": "yellow", "skipped": "dim"}[r.status]
        lat = f" [dim]{r.latency_ms}ms[/]" if r.latency_ms is not None else ""
        from rich.markup import escape as _esc
        extra = f" — {_esc(r.error)}" if r.error and r.status != "ok" else ""
        sample = f"  [dim]{_esc(repr(r.sample))}[/]" if r.sample else ""
        console.print(f"[{color}]{r.emoji}[/] [{color}]{r.status:7}[/]{lat}  {_esc(r.tinker_path)}{sample}{extra}")


# ---------- use / active ----------
@app.command("use")
def use(path: str = typer.Argument(..., help="tinker:// path to mark active.")):
    """Mark a checkpoint as the active one. Other tools can source ~/.tinkpad/active.env."""
    if not path.startswith("tinker://"):
        console.print("[red]not a tinker:// path[/]")
        raise typer.Exit(2)
    p = active_mod.set_active(path)
    console.print(f"[green]✓[/] active set: [bold]{path}[/]")
    console.print(f"[dim]wrote {p} and {p.parent / 'active.env'}[/]")
    console.print(f"[dim]to use: `source ~/.tinkpad/active.env`[/]")


@app.command("active")
def active(clear: bool = typer.Option(False, "--clear", "-c")):
    """Show (or clear) the active checkpoint."""
    if clear:
        active_mod.clear_active()
        console.print("[yellow]cleared.[/]")
        return
    a = active_mod.get_active()
    if not a:
        console.print("[dim](none)[/]")
        raise typer.Exit(1)
    console.print(a)


# ---------- registry ----------
reg_app = typer.Typer(help="Manage local run_id → experiment-name registry.", no_args_is_help=True)
app.add_typer(reg_app, name="reg")


@reg_app.command("list")
def reg_list():
    """Show the registry."""
    reg = Registry()
    if not reg.all():
        console.print("[dim](registry empty — try `tinkpad reg scan`)[/]")
        return
    for rid, e in sorted(reg.all().items(), key=lambda kv: kv[1].name):
        src = f"  [dim]{e.source_path}[/]" if e.source_path else ""
        console.print(f"[bold]{e.name:30}[/] {short_run(rid):10} [dim]{rid}[/]{src}")


@reg_app.command("set")
def reg_set(run_id: str, name: str, note: Optional[str] = typer.Option(None, "--note")):
    """Manually map a run_id to an experiment name."""
    reg = Registry()
    reg.set(run_id, name, note=note)
    reg.save()
    console.print(f"[green]✓[/] {short_run(run_id)} → [bold]{name}[/]")


@reg_app.command("rm")
def reg_rm(run_id: str):
    reg = Registry()
    if reg.remove(run_id):
        reg.save()
        console.print(f"[yellow]removed[/] {short_run(run_id)}")
    else:
        console.print(f"[red]not found:[/] {run_id}")
        raise typer.Exit(1)


@reg_app.command("scan")
def reg_scan(
    root: list[Path] = typer.Option(None, "--root", help="Override scan roots (repeatable)."),
    save: bool = typer.Option(False, "--save", help="Persist these roots as the default."),
):
    """Walk local folders for Zlog/<run_id> entries and auto-register."""
    roots = root if root else load_scan_roots()
    if save and root:
        save_scan_roots(root)
        console.print(f"[dim]saved scan roots: {root}[/]")
    found = scan(roots)
    console.print(f"[green]scanned[/] {len(roots)} root(s); registered/refreshed {len(found)} entries")
    if found:
        for e in found[:20]:
            console.print(f"  [bold]{e.name:30}[/] [dim]{e.run_id}[/]")
        if len(found) > 20:
            console.print(f"  [dim]... and {len(found) - 20} more[/]")


@reg_app.command("roots")
def reg_roots(add: list[Path] = typer.Option(None, "--add"), clear: bool = typer.Option(False)):
    if clear:
        save_scan_roots([])
        console.print("[yellow]cleared scan roots[/]")
        return
    roots = load_scan_roots()
    if add:
        roots = list({*roots, *add})
        save_scan_roots(roots)
    for r in roots:
        console.print(str(r))


# ---------- tui ----------
@app.command("tui")
def tui_cmd():
    """Launch the interactive TUI browser."""
    from .tui import run as tui_run
    tui_run()


# ---------- doctor ----------
@app.command("doctor")
def doctor():
    """Sanity-check environment."""
    from . import config
    issues = 0
    console.print(f"tinkpad {__version__}")
    if config.api_key():
        console.print("[green]✓[/] TINKER_API_KEY present")
    else:
        console.print("[red]✗[/] TINKER_API_KEY not set"); issues += 1
    try:
        client = TinkerClient()
        runs = client.list_runs()
        console.print(f"[green]✓[/] API reachable — {len(runs)} runs visible")
    except Exception as e:
        console.print(f"[red]✗[/] API call failed: {e}"); issues += 1
    console.print(f"[dim]registry:[/] {config.REGISTRY_PATH}")
    console.print(f"[dim]active:  [/] {config.ACTIVE_PATH} {'(set)' if active_mod.get_active() else '(unset)'}")
    if issues:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
