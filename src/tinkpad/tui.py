"""Interactive Textual TUI.

Layout:
  ┌── runs (left, ~40%) ──────┬── checkpoints for selected run (right) ──┐
  │ ▸ experiment-name  age    │ checkpoint_id  type  size  age  probe    │
  │ ...                       │ ...                                       │
  └───────────────────────────┴───────────────────────────────────────────┘
  status bar: active checkpoint • last action

Keys:
  enter / space   probe selected checkpoint
  u               mark selected as active (use)
  r               refresh from API
  /               focus search
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
from .formatting import human_age, human_size, short_run
from .probe import probe as probe_one
from .registry import Registry, scan, load_scan_roots
from .tinker_client import Checkpoint, Run, TinkerClient

from .config import api_key


PROBE_COLORS = {"ok": "green", "fail": "red", "timeout": "yellow", "skipped": "dim"}


class TinkpadApp(App):
    CSS = """
    Screen { layout: vertical; }
    #panes { height: 1fr; }
    #runs-table, #ckpts-table { height: 1fr; }
    #status { dock: bottom; height: 1; padding: 0 1; background: $boost; }
    Input { dock: bottom; height: 3; }
    .pane { border: round $primary; padding: 0 1; }
    """

    BINDINGS = [
        Binding("q", "quit", "Quit"),
        Binding("r", "refresh", "Refresh"),
        Binding("u", "use", "Use (active)"),
        Binding("p,enter,space", "probe", "Probe"),
        Binding("a", "probe_all_in_run", "Probe-all-in-run"),
        Binding("slash", "search", "Search"),
        Binding("escape", "blur_search", show=False),
    ]

    def __init__(self):
        super().__init__()
        self.client: TinkerClient | None = None
        self.reg: Registry | None = None
        self.runs: list[Run] = []
        self.ckpts_by_run: dict[str, list[Checkpoint]] = {}
        self.probe_status: dict[str, tuple[str, int | None]] = {}
        self.filter_text: str = ""

    # ---------- compose ----------
    def compose(self) -> ComposeResult:
        yield Header(show_clock=True)
        with Vertical(id="panes"):
            with Horizontal():
                yield DataTable(id="runs-table", classes="pane", zebra_stripes=True, cursor_type="row")
                yield DataTable(id="ckpts-table", classes="pane", zebra_stripes=True, cursor_type="row")
        yield Input(placeholder="search experiment/run/checkpoint…", id="search", classes="hidden")
        yield Static("", id="status")
        yield Footer()

    # ---------- lifecycle ----------
    async def on_mount(self) -> None:
        runs_t = self.query_one("#runs-table", DataTable)
        runs_t.add_columns("experiment", "run", "model", "last activity")
        ckpts_t = self.query_one("#ckpts-table", DataTable)
        ckpts_t.add_columns("checkpoint", "type", "size", "age", "probe")
        self.query_one("#search", Input).display = False
        await self._refresh()

    async def _refresh(self) -> None:
        self._status("loading…")
        scan(load_scan_roots())
        self.reg = Registry()
        try:
            self.client = self.client or TinkerClient()
            self.runs = await asyncio.to_thread(self.client.list_runs)
        except Exception as e:
            self._status(f"[red]error: {e}[/]")
            return
        # Pre-bucket checkpoints lazily; only fetch when selected.
        self.ckpts_by_run.clear()
        self._render_runs()
        self._status(f"loaded {len(self.runs)} runs")

    # ---------- rendering ----------
    def _filter_match(self, run: Run) -> bool:
        if not self.filter_text:
            return True
        f = self.filter_text.lower()
        name = (self.reg.name_for(run.run_id) or "").lower()
        return f in name or f in run.run_id.lower() or f in run.base_model.lower()

    def _render_runs(self) -> None:
        t = self.query_one("#runs-table", DataTable)
        t.clear()
        for r in self.runs:
            if not self._filter_match(r):
                continue
            name = self.reg.name_for(r.run_id) or "—"
            style = "red" if r.corrupted else ""
            t.add_row(
                f"[{style}]{name}[/]" if style else name,
                short_run(r.run_id),
                r.base_model,
                human_age(r.last_request_time),
                key=r.run_id,
            )
        if t.row_count:
            t.move_cursor(row=0)
            self.call_after_refresh(self._render_ckpts_for_cursor)

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
        ckpts = self.ckpts_by_run.get(run_id)
        if ckpts is None:
            self._status(f"loading checkpoints for {short_run(run_id)}…")

            async def _load():
                try:
                    cs = await asyncio.to_thread(self.client.list_checkpoints, run_id)
                except Exception as e:
                    self._status(f"[red]error: {e}[/]")
                    return
                self.ckpts_by_run[run_id] = cs
                # only rerender if cursor still on this run
                runs_t = self.query_one("#runs-table", DataTable)
                try:
                    cur = runs_t.coordinate_to_cell_key(runs_t.cursor_coordinate).row_key.value
                except Exception:
                    cur = None
                if cur == run_id:
                    self._render_ckpts(run_id)
                self._status(f"{len(cs)} checkpoints loaded for {short_run(run_id)}")

            self.run_worker(_load(), exclusive=False)
            return
        for c in ckpts:
            ps = self.probe_status.get(c.tinker_path)
            if ps:
                status, lat = ps
                color = PROBE_COLORS.get(status, "white")
                emoji = {"ok": "✓", "fail": "✗", "timeout": "⌛", "skipped": "-"}[status]
                lat_s = f" {lat}ms" if lat is not None else ""
                probe_cell = f"[{color}]{emoji} {status}{lat_s}[/]"
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
        if event.input.id == "search":
            self.filter_text = event.value
            self._render_runs()

    def on_input_submitted(self, event: Input.Submitted) -> None:
        if event.input.id == "search":
            event.input.display = False
            self.query_one("#runs-table", DataTable).focus()

    # ---------- actions ----------
    def action_search(self) -> None:
        i = self.query_one("#search", Input)
        i.display = True
        i.focus()

    def action_blur_search(self) -> None:
        i = self.query_one("#search", Input)
        if i.display:
            i.display = False
            self.query_one("#runs-table", DataTable).focus()

    async def action_refresh(self) -> None:
        await self._refresh()

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
        # Mark as in-flight
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
