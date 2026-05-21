"""Interactive Textual TUI.

Layout:
  ┌── runs (left, ~40%) ──────┬── checkpoints for selected run (right) ──┐
  │ ▸ experiment-name  age    │ checkpoint_id  type  size  age  probe    │
  │ ...                       │ ...                                       │
  └───────────────────────────┴───────────────────────────────────────────┘
  status bar: active checkpoint • last action

Keys:
  ←  / →          switch focus between runs pane and checkpoints pane
  ↑  / ↓          move cursor within current pane
  enter / space   probe selected checkpoint
  n               rename selected run (inline)
  u               mark selected checkpoint as active
  p               probe selected checkpoint
  a               probe every sampler in current run
  r               refresh from API
  s               force-sync the cache
  /               search
  q               quit
"""
from __future__ import annotations

import asyncio
from typing import Iterable

from textual.app import App, ComposeResult
from textual.binding import Binding
from textual.containers import Horizontal, Vertical
from textual.widgets import DataTable, Footer, Header, Input, Static

from . import active as active_mod
from . import cache as cache_mod
from .formatting import human_age, human_size, short_run
from .probe import probe as probe_one
from .registry import Registry, scan, load_scan_roots
from .tinker_client import Checkpoint, Run, TinkerClient

from .config import api_key


PROBE_COLORS = {"ok": "green", "fail": "red", "timeout": "yellow", "skipped": "dim", "…": "cyan"}
PROBE_GLYPHS = {"ok": "✓", "fail": "✗", "timeout": "⌛", "skipped": "-", "…": "…"}
UNNAMED_LABEL = "[red dim]\\[unnamed][/]"


class TinkpadApp(App):
    CSS = """
    Screen { layout: vertical; }
    #panes { height: 1fr; }
    #runs-table, #ckpts-table { height: 1fr; }
    #runs-table { width: 40%; }
    #status { dock: bottom; height: 1; padding: 0 1; background: $boost; }
    Input { dock: bottom; height: 3; }
    .pane { border: round $primary-darken-2; padding: 0 1; }
    .focused { border: round $accent; }
    .hidden { display: none; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("s", "sync", "Sync"),
        Binding("u", "use", "Use"),
        Binding("p,enter,space", "probe", "Probe"),
        Binding("a", "probe_all_in_run", "Probe-run"),
        Binding("n", "rename", "Rename"),
        Binding("left", "focus_runs", "← runs"),
        Binding("right", "focus_ckpts", "ckpts →"),
        Binding("slash", "search", "Search"),
        Binding("escape", "blur_input", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.client: TinkerClient | None = None
        self.reg: Registry | None = None
        self.runs: list[Run] = []
        self.all_ckpts: list[Checkpoint] = []  # full cache
        self.ckpts_by_run: dict[str, list[Checkpoint]] = {}
        self.probe_status: dict[str, tuple[str, int | None]] = {}
        self.filter_text: str = ""
        self._input_mode: str | None = None  # "search" | "rename" | None
        self._rename_target_run: str | None = None

    # ---------- compose ----------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="panes"):
            with Horizontal():
                yield DataTable(id="runs-table", classes="pane focused", zebra_stripes=True, cursor_type="row")
                yield DataTable(id="ckpts-table", classes="pane", zebra_stripes=True, cursor_type="row")
        yield Input(placeholder="…", id="input", classes="hidden")
        yield Static("", id="status")
        yield Footer()

    # ---------- lifecycle ----------
    async def on_mount(self) -> None:
        runs_t = self.query_one("#runs-table", DataTable)
        runs_t.add_columns("experiment", "run", "model", "last activity", "ckpt created")
        ckpts_t = self.query_one("#ckpts-table", DataTable)
        ckpts_t.add_columns("checkpoint", "type", "size", "age", "probe")
        self.query_one("#input", Input).display = False
        await self._load_from_cache_then_refresh()
        runs_t.focus()

    async def _load_from_cache_then_refresh(self) -> None:
        # Cheap: paint from cache immediately.
        runs, ckpts, ts = cache_mod.load()
        self.reg = Registry()
        if runs:
            self.runs = runs
            self.all_ckpts = ckpts
            self._bucket_ckpts()
            self._render_runs()
            self._status(f"loaded {len(runs)} runs from cache")
        # Then refresh if stale.
        if not cache_mod.is_fresh():
            await self._sync(silent=True)

    async def _sync(self, silent: bool = False) -> None:
        if not silent:
            self._status("syncing…")
        try:
            self.client = self.client or TinkerClient()
            await asyncio.to_thread(cache_mod.sync, self.client)
        except Exception as e:
            self._status(f"[red]sync error: {e}[/]")
            return
        runs, ckpts, _ = cache_mod.load()
        self.runs = runs
        self.all_ckpts = ckpts
        self._bucket_ckpts()
        scan(load_scan_roots())  # opportunistic
        self.reg = Registry()
        self._render_runs()
        self._status(f"synced — {len(runs)} runs, {len(ckpts)} checkpoints")

    def _bucket_ckpts(self) -> None:
        self.ckpts_by_run.clear()
        for c in self.all_ckpts:
            self.ckpts_by_run.setdefault(c.run_id, []).append(c)
        for k in self.ckpts_by_run:
            self.ckpts_by_run[k].sort(key=lambda c: c.created_at or 0, reverse=True)

    # ---------- rendering ----------
    def _filter_match(self, run: Run) -> bool:
        if not self.filter_text:
            return True
        f = self.filter_text.lower()
        name = (self.reg.name_for(run.run_id) or "").lower()
        return f in name or f in run.run_id.lower() or f in run.base_model.lower()

    def _render_runs(self) -> None:
        t = self.query_one("#runs-table", DataTable)
        cursor_row_key = None
        try:
            cursor_row_key = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        except Exception:
            pass
        t.clear()
        unnamed = 0
        new_index = 0
        target_index = 0
        for i, r in enumerate(self.runs):
            if not self._filter_match(r):
                continue
            name = self.reg.name_for(r.run_id)
            if name is None:
                unnamed += 1
                label = UNNAMED_LABEL
            else:
                label = f"[bold]{name}[/]" if not r.corrupted else f"[red bold]{name}[/]"
            ckpt_t = r.last_sampler_checkpoint_created_at or r.last_checkpoint_created_at
            t.add_row(
                label,
                short_run(r.run_id),
                r.base_model,
                human_age(r.last_request_time),
                human_age(ckpt_t) if ckpt_t else "—",
                key=r.run_id,
            )
            if cursor_row_key == r.run_id:
                target_index = new_index
            new_index += 1
        if t.row_count:
            t.move_cursor(row=target_index)
            self.call_after_refresh(self._render_ckpts_for_cursor)
        active = active_mod.get_active()
        unnamed_s = f" — [red]{unnamed} unnamed[/]" if unnamed else ""
        active_s = f"active: [bold]{active}[/]   " if active else ""
        self.query_one("#status", Static).update(f"{active_s}{len(self.runs)} runs{unnamed_s}")

    def _render_ckpts_for_cursor(self) -> None:
        runs_t = self.query_one("#runs-table", DataTable)
        if runs_t.row_count == 0:
            self.query_one("#ckpts-table", DataTable).clear()
            return
        try:
            run_id = runs_t.coordinate_to_cell_key(runs_t.cursor_coordinate).row_key.value
        except Exception:
            return
        self._render_ckpts(run_id)

    def _render_ckpts(self, run_id: str) -> None:
        t = self.query_one("#ckpts-table", DataTable)
        t.clear()
        ckpts = self.ckpts_by_run.get(run_id, [])
        for c in ckpts:
            ps = self.probe_status.get(c.tinker_path)
            if ps:
                status, lat = ps
                color = PROBE_COLORS.get(status, "white")
                glyph = PROBE_GLYPHS.get(status, "?")
                lat_s = f" {lat}ms" if lat is not None else ""
                probe_cell = f"[{color}]{glyph} {status}{lat_s}[/]"
            else:
                probe_cell = "[dim]—[/]"
            type_color = "magenta" if c.type == "sampler" else "yellow"
            t.add_row(
                c.checkpoint_id,
                f"[{type_color}]{c.type}[/]",
                human_size(c.size_bytes),
                human_age(c.created_at),
                probe_cell,
                key=c.tinker_path,
            )

    def _status(self, msg: str) -> None:
        active = active_mod.get_active()
        prefix = f"active: [bold]{active}[/]   " if active else ""
        self.query_one("#status", Static).update(prefix + msg)

    # ---------- events ----------
    def on_data_table_row_highlighted(self, event: DataTable.RowHighlighted) -> None:
        if event.data_table.id == "runs-table":
            run_id = event.row_key.value
            self._render_ckpts(run_id)

    def on_input_changed(self, event: Input.Changed) -> None:
        if self._input_mode == "search":
            self.filter_text = event.value
            self._render_runs()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        i = self.query_one("#input", Input)
        if self._input_mode == "search":
            self._close_input()
            self.query_one("#runs-table", DataTable).focus()
        elif self._input_mode == "rename":
            new_name = event.value.strip()
            if new_name and self._rename_target_run:
                self.reg.set(self._rename_target_run, new_name)
                self.reg.save()
                self._status(f"[green]renamed[/] → [bold]{new_name}[/]")
            self._close_input()
            self._render_runs()
            self.query_one("#runs-table", DataTable).focus()

    def _close_input(self) -> None:
        i = self.query_one("#input", Input)
        i.display = False
        i.value = ""
        self._input_mode = None
        self._rename_target_run = None

    # ---------- actions ----------
    def _focus_runs(self) -> None:
        self.query_one("#runs-table", DataTable).focus()
        self.query_one("#runs-table").add_class("focused")
        self.query_one("#ckpts-table").remove_class("focused")

    def _focus_ckpts(self) -> None:
        self.query_one("#ckpts-table", DataTable).focus()
        self.query_one("#ckpts-table").add_class("focused")
        self.query_one("#runs-table").remove_class("focused")

    def action_focus_runs(self) -> None:
        self._focus_runs()

    def action_focus_ckpts(self) -> None:
        self._focus_ckpts()

    def action_search(self) -> None:
        i = self.query_one("#input", Input)
        i.placeholder = "search experiment/run/model…"
        i.value = self.filter_text
        i.display = True
        self._input_mode = "search"
        i.focus()

    def action_rename(self) -> None:
        runs_t = self.query_one("#runs-table", DataTable)
        try:
            run_id = runs_t.coordinate_to_cell_key(runs_t.cursor_coordinate).row_key.value
        except Exception:
            return
        current = self.reg.name_for(run_id) or ""
        i = self.query_one("#input", Input)
        i.placeholder = f"rename run {short_run(run_id)} → "
        i.value = current
        i.display = True
        self._input_mode = "rename"
        self._rename_target_run = run_id
        i.focus()

    def action_blur_input(self) -> None:
        if self._input_mode:
            self._close_input()
            self.query_one("#runs-table", DataTable).focus()

    async def action_refresh(self) -> None:
        await self._load_from_cache_then_refresh()

    async def action_sync(self) -> None:
        await self._sync()

    def _selected_ckpt(self) -> Checkpoint | None:
        t = self.query_one("#ckpts-table", DataTable)
        if t.row_count == 0:
            return None
        try:
            path = t.coordinate_to_cell_key(t.cursor_coordinate).row_key.value
        except Exception:
            return None
        runs_t = self.query_one("#runs-table", DataTable)
        try:
            run_id = runs_t.coordinate_to_cell_key(runs_t.cursor_coordinate).row_key.value
        except Exception:
            return None
        for c in self.ckpts_by_run.get(run_id, []):
            if c.tinker_path == path:
                return c
        return None

    def action_use(self) -> None:
        c = self._selected_ckpt()
        if not c:
            self._status("[yellow]no checkpoint selected[/]")
            return
        active_mod.set_active(c.tinker_path)
        self._status(f"[green]active set:[/] {c.tinker_path}")

    def action_probe(self) -> None:
        c = self._selected_ckpt()
        if not c:
            self._status("[yellow]no checkpoint selected[/]")
            return
        self._spawn_probe([c])

    def action_probe_all_in_run(self) -> None:
        runs_t = self.query_one("#runs-table", DataTable)
        try:
            run_id = runs_t.coordinate_to_cell_key(runs_t.cursor_coordinate).row_key.value
        except Exception:
            return
        ckpts = [c for c in self.ckpts_by_run.get(run_id, []) if c.type == "sampler"]
        if not ckpts:
            self._status("[yellow]no sampler checkpoints in this run[/]")
            return
        self._spawn_probe(ckpts)

    def _spawn_probe(self, ckpts: Iterable[Checkpoint]) -> None:
        ckpts = list(ckpts)
        key = api_key()
        if not key:
            self._status("[red]TINKER_API_KEY not set[/]")
            return
        for c in ckpts:
            self.probe_status[c.tinker_path] = ("…", None)
        self._render_ckpts_for_cursor()

        sem = asyncio.Semaphore(4)

        async def _one(c: Checkpoint):
            async with sem:
                r = await asyncio.to_thread(probe_one, c.tinker_path)
            self.probe_status[c.tinker_path] = (r.status, r.latency_ms)
            self._render_ckpts_for_cursor()

        async def _go():
            await asyncio.gather(*[_one(c) for c in ckpts])
            self._status(f"probed {len(ckpts)} checkpoint(s)")

        self.run_worker(_go(), exclusive=False)


def run() -> None:
    TinkpadApp().run()
