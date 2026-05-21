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
from . import cache as cache_mod
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
    help="tinkpad — browse / probe / copy Tinker checkpoints.",
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


def _cached_or_live(force_fresh: bool = False) -> tuple[list, list]:
    """Get runs+checkpoints from cache, refreshing if stale.

    Used by read-only commands (ls/runs/tree). Falls back to whatever's on
    disk if the API is unreachable.
    """
    client = _client() if (force_fresh or not cache_mod.is_fresh()) else None
    return cache_mod.load_or_sync(client, force=force_fresh)


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
      - '@latest':          most-recently-created sampler across all runs

    Raises typer.Exit on 0/ambiguous matches.
    """
    if query.startswith("tinker://"):
        return query
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
    runs_list, ckpts_list = _cached_or_live()

    runs_by_short = {r.run_id.split(":", 1)[0]: r for r in runs_list}
    if run:
        keep = set()
        for rid_short, r in runs_by_short.items():
            name = reg.name_for(r.run_id) or ""
            if run in r.run_id or (name and run in name):
                keep.add(rid_short)
        runs_by_short = {k: v for k, v in runs_by_short.items() if k in keep}

    all_ckpts = ckpts_list
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


@app.command("runs")
def runs(fresh: bool = typer.Option(False, "--fresh", help="Force-refresh from API.")):
    """List training runs with last-activity and last-sampler info."""
    maybe_scan(load_scan_roots())
    reg = Registry()
    rs, _ = _cached_or_live(force_fresh=fresh)
    unnamed = sum(1 for r in rs if reg.name_for(r.run_id) is None)
    console.print(runs_table(rs, _name_for(reg)))
    if unnamed:
        console.print(
            f"[red]{unnamed} unnamed run(s)[/] — give them names with [bold]tinkpad reg name-unnamed[/] or [bold]tinkpad tui[/] (press n)."
        )


@app.command("sync")
def sync_cmd():
    """Refresh the local metadata cache from the live API."""
    client = _client()
    n_runs, n_ckpts = cache_mod.sync(client)
    console.print(f"[green]synced[/] — {n_runs} runs, {n_ckpts} checkpoints cached at {cache_mod.CACHE_PATH}")


@app.command("tree")
def tree_cmd(
    run: Optional[str] = typer.Option(None, "--run", "-r"),
    sampler_only: bool = typer.Option(True, "--sampler/--all-types", "-s"),
    fresh: bool = typer.Option(False, "--fresh"),
):
    """File-system-style view: experiment / run / checkpoint."""
    from rich.tree import Tree

    maybe_scan(load_scan_roots())
    reg = Registry()
    rs, all_ckpts = _cached_or_live(force_fresh=fresh)

    ckpts_by_short: dict[str, list[Checkpoint]] = {}
    for c in all_ckpts:
        if sampler_only and c.type != "sampler":
            continue
        ckpts_by_short.setdefault(c.run_id.split(":", 1)[0], []).append(c)
    for k, v in ckpts_by_short.items():
        v.sort(key=lambda c: c.created_at or 0, reverse=True)

    # Group runs by experiment name. Each unnamed run becomes its own bucket
    # so they don't all collapse together.
    by_exp: dict[str, list] = {}
    for r in rs:
        name = reg.name_for(r.run_id)
        if name is None:
            key = f"__unnamed__/{short_run(r.run_id)}"
        else:
            key = name
        by_exp.setdefault(key, []).append(r)

    if run:
        ql = run.lower()
        by_exp = {
            k: [r for r in v if (ql in r.run_id.lower() or (reg.name_for(r.run_id) and ql in reg.name_for(r.run_id).lower()))]
            for k, v in by_exp.items()
        }
        by_exp = {k: v for k, v in by_exp.items() if v}

    def _label_for(key: str) -> str:
        if key.startswith("__unnamed__/"):
            short = key.split("/", 1)[1]
            return f"📁 [red dim]\\[unnamed][/] [dim]{short}[/]"
        return f"📁 [bold]{key}[/]"

    root = Tree("[bold cyan]tinkpad[/]  [dim]experiments → runs → checkpoints[/]")
    # Named first (alphabetical), unnamed last (by short id).
    named_keys = sorted([k for k in by_exp if not k.startswith("__unnamed__/")], key=str.lower)
    unnamed_keys = sorted([k for k in by_exp if k.startswith("__unnamed__/")])
    for exp_name in named_keys + unnamed_keys:
        runs_for = by_exp[exp_name]
        exp_node = root.add(_label_for(exp_name))
        for r in runs_for:
            run_label = f"🏃 [dim]{short_run(r.run_id)}[/]  [italic]{r.base_model}[/]  [dim]{human_age(r.last_request_time)}[/]"
            if r.corrupted:
                run_label += " [red](corrupted)[/]"
            run_node = exp_node.add(run_label)
            short = r.run_id.split(":", 1)[0]
            ckpts = ckpts_by_short.get(short, [])
            if not ckpts:
                run_node.add("[dim](no checkpoints)[/]")
                continue
            for c in ckpts:
                glyph = "📦" if c.type == "sampler" else "🧱"
                t_color = "magenta" if c.type == "sampler" else "yellow"
                run_node.add(
                    f"{glyph} {c.checkpoint_id}  [{t_color}]{c.type}[/]  [dim]{human_size(c.size_bytes)}  {human_age(c.created_at)}[/]"
                )
    console.print(root)


# ---------- info ----------
@app.command("info")
def info(path: str = typer.Argument(..., help="Full URI, short like '5a2c6:final', or '@latest'.")):
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
    paths: list[str] = typer.Argument(None, help="One or more paths (full URI, short like '5a2c6:final', '@latest')."),
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
        console.print("[red]no path given[/]")
        raise typer.Exit(2)

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


@reg_app.command("name-unnamed")
def reg_name_unnamed():
    """Walk every unnamed run and prompt for a name. Blank = skip."""
    rs, _ = _cached_or_live()
    reg = Registry()
    unnamed = [r for r in rs if reg.name_for(r.run_id) is None]
    if not unnamed:
        console.print("[green]all runs already have names[/]")
        return
    console.print(f"[bold]{len(unnamed)} unnamed run(s)[/]  [dim](blank = skip, ctrl-c = stop)[/]\n")
    named = 0
    for i, r in enumerate(unnamed, 1):
        last_sk = r.last_sampler_checkpoint_path.split("/")[-1] if r.last_sampler_checkpoint_path else "—"
        console.print(
            f"[bold]({i}/{len(unnamed)})[/]  {short_run(r.run_id)}  "
            f"[italic]{r.base_model}[/]  [dim]{human_age(r.last_request_time)}  last sampler: {last_sk}[/]"
        )
        try:
            new_name = typer.prompt("  name", default="", show_default=False)
        except (KeyboardInterrupt, typer.Abort):
            console.print("\n[yellow]stopped[/]")
            break
        new_name = new_name.strip()
        if new_name:
            reg.set(r.run_id, new_name)
            reg.save()
            named += 1
            console.print(f"  [green]✓ {new_name}[/]\n")
        else:
            console.print("  [dim]skipped[/]\n")
    console.print(f"[green]named {named} run(s)[/]")


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
    console.print(f"[dim]cache:   [/] {cache_mod.CACHE_PATH}")
    if issues:
        raise typer.Exit(1)


if __name__ == "__main__":
    app()
