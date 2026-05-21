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
from .registry import Registry, maybe_scan, scan, load_scan_roots, save_scan_roots
from .tinker_client import Checkpoint, TinkerClient

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


def _resolve_path(client: TinkerClient, query: str) -> str:
    """Resolve a path-ish query to a full tinker:// checkpoint URI.

    Accepted forms:
      - full URI:           tinker://abc...:train:0/sampler_weights/000010
      - short run + step:   5a2c6:000010   |   5a2c6:final   |   myexp:final
      - short run only:     5a2c6   (→ latest sampler for that run)
      - experiment name:    myexp           (→ latest sampler for matching run)
      - 'active':           the active checkpoint
      - '@latest':          most-recently-created sampler across all runs

    Raises typer.Exit on 0/ambiguous matches.
    """
    if query.startswith("tinker://"):
        return query
    if query == "active":
        a = active_mod.get_active()
        if not a:
            console.print("[red]no active checkpoint set[/]")
            raise typer.Exit(1)
        return a
    if query == "@latest":
        ckpts = [c for c in client.list_checkpoints() if c.type == "sampler" and c.created_at]
        if not ckpts:
            console.print("[red]no sampler checkpoints found[/]")
            raise typer.Exit(1)
        return max(ckpts, key=lambda c: c.created_at).tinker_path

    run_part, _, step_part = query.partition(":")
    # If the user wrote "5a2c64eb:train:0/sampler_weights/final" that's an
    # un-prefixed full path — re-attach tinker:// and return.
    if "/" in step_part:
        return f"tinker://{query}"

    full_rid = _resolve_run_id(client, run_part)
    ckpts = [c for c in client.list_checkpoints(full_rid) if c.type == "sampler"]
    if not ckpts:
        console.print(f"[red]no sampler checkpoints for run {short_run(full_rid)}[/]")
        raise typer.Exit(1)

    if not step_part or step_part == "latest":
        # Latest = most recently created.
        return max(ckpts, key=lambda c: c.created_at or 0).tinker_path

    # Match a checkpoint by step suffix, e.g. "final", "30", "000030".
    norm_step = step_part.lstrip("0") or "0"
    matches = []
    for c in ckpts:
        step = c.checkpoint_id.split("/")[-1]
        if step == step_part or step.lstrip("0") == norm_step:
            matches.append(c)
    if len(matches) == 1:
        return matches[0].tinker_path
    if not matches:
        console.print(f"[red]no step '{step_part}' in run {short_run(full_rid)}[/]")
        console.print("[dim]available: " + ", ".join(sorted({c.checkpoint_id.split('/')[-1] for c in ckpts})) + "[/]")
        raise typer.Exit(1)
    console.print(f"[red]ambiguous step '{step_part}':[/]")
    for c in matches:
        console.print(f"  {c.tinker_path}")
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
    per_run: int = typer.Option(3, "--per-run", "-k", help="Show only this many checkpoints per run (newest first)."),
    show_all: bool = typer.Option(False, "--all", "-A", help="Disable per-run grouping; flat list."),
    limit: int = typer.Option(500, "--limit", "-n"),
):
    """List checkpoints across all runs, joined with local experiment names.

    By default, groups by run and shows the newest few checkpoints per run.
    Use `--all` for a flat list (the old behavior).
    """
    maybe_scan(load_scan_roots())
    reg = Registry()
    client = _client()

    runs_by_short = {r.run_id.split(":", 1)[0]: r for r in client.list_runs()}
    if run:
        keep = set()
        for rid_short, r in runs_by_short.items():
            name = reg.name_for(r.run_id) or ""
            if run in r.run_id or (name and run in name):
                keep.add(rid_short)
        runs_by_short = {k: v for k, v in runs_by_short.items() if k in keep}

    if run and len(runs_by_short) == 1:
        rid = next(iter(runs_by_short.values())).run_id
        all_ckpts = client.list_checkpoints(rid)
    else:
        all_ckpts = client.list_checkpoints()
        if run:
            keep_ids = set(runs_by_short.keys())
            all_ckpts = [c for c in all_ckpts if c.run_id.split(":", 1)[0] in keep_ids]

    if sampler_only:
        all_ckpts = [c for c in all_ckpts if c.type == "sampler"]
    if training_only:
        all_ckpts = [c for c in all_ckpts if c.type == "training"]

    # Group + trim newest-first per run.
    if not show_all:
        grouped: dict[str, list[Checkpoint]] = {}
        for c in all_ckpts:
            grouped.setdefault(c.run_id.split(":", 1)[0], []).append(c)
        trimmed: list[Checkpoint] = []
        for rid, cs in grouped.items():
            cs.sort(key=lambda c: c.created_at or 0, reverse=True)
            trimmed.extend(cs[:per_run])
        # Order groups by their newest checkpoint
        trimmed.sort(key=lambda c: (c.run_id, -((c.created_at.timestamp() if c.created_at else 0))))
        all_ckpts = trimmed[:limit]
    else:
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
    if not show_all:
        seen_runs = {c.run_id.split(":", 1)[0] for c in all_ckpts}
        console.print(
            f"[dim]showing newest {per_run} per run across {len(seen_runs)} run(s); use --all for full list[/]"
        )
    active = active_mod.get_active()
    if active:
        console.print(f"[dim]active:[/] [bold]{active}[/]")


@app.command("runs")
def runs():
    """List training runs with last-activity and last-sampler info."""
    maybe_scan(load_scan_roots())
    reg = Registry()
    client = _client()
    rs = client.list_runs()
    console.print(runs_table(rs, _name_for(reg)))


# ---------- info ----------
@app.command("info")
def info(path: str = typer.Argument(..., help="Full URI, short like '5a2c6:final', 'active', or '@latest'.")):
    """Detail view of one checkpoint (with probe)."""
    reg = Registry()
    client = _client()
    path = _resolve_path(client, path)
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
    console.print(Panel("\n".join(body), title=match.checkpoint_id, border_style="cyan"), emoji=False)

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
    paths: list[str] = typer.Argument(None, help="One or more paths (full URI, short like '5a2c6:final', 'active', '@latest')."),
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
        target_paths = [_resolve_path(client, p) for p in paths]
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
def use(
    path: str = typer.Argument(..., help="Full URI, short like '5a2c6:final', or '@latest'."),
    no_verify: bool = typer.Option(False, "--no-verify", help="Skip the pre-flight probe."),
):
    """Mark a checkpoint as the active one. Other tools can source ~/.tinkpad/active.env."""
    client = _client()
    resolved = _resolve_path(client, path)
    if not no_verify:
        console.print(f"[dim]probing {resolved}…[/]")
        res = probe_one(resolved)
        if res.status != "ok":
            color = {"fail": "red", "timeout": "yellow", "skipped": "yellow"}.get(res.status, "red")
            console.print(f"[{color}]✗ probe {res.status}[/] — {res.error or 'checkpoint may not serve'}")
            console.print("[dim]pass --no-verify to set anyway[/]")
            raise typer.Exit(1)
        console.print(f"[green]✓ probe ok[/] ({res.latency_ms}ms)")
    p = active_mod.set_active(resolved)
    console.print(f"[green]✓[/] active set: [bold]{resolved}[/]")
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
    verbose: bool = typer.Option(False, "--verbose", "-v"),
):
    """Walk local folders for Zlog/<run_id> entries and auto-register."""
    roots = root if root else load_scan_roots()
    if save and root:
        save_scan_roots(root)
        console.print(f"[dim]saved scan roots: {root}[/]")
    for r in roots:
        marker = "[green]ok[/]" if r.exists() else "[red]missing[/]"
        console.print(f"[dim]scanning:[/] {r}  {marker}")
    found = scan(roots)
    console.print(f"[green]scanned[/] {len(roots)} root(s); registered/refreshed {len(found)} entries")
    if not found:
        console.print(
            "[dim]No matches. tinkpad looks for either:\n"
            "  • a folder named <run-uuid>\n"
            "  • a folder containing Zlog/<run-uuid>/...\n"
            "Use `tinkpad reg set <run-id> <name>` to map manually.[/]"
        )
        return
    for e in found[:20] if not verbose else found:
        src = f"  [dim]({e.source_path})[/]" if e.source_path else ""
        console.print(f"  [bold]{e.name:30}[/] [dim]{e.run_id}[/]{src}")
    if not verbose and len(found) > 20:
        console.print(f"  [dim]... and {len(found) - 20} more (use -v to see all)[/]")


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
