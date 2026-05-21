# tinkpad

> A beautifully-designed wrapper around [Tinker](https://thinkingmachines.ai/tinker/) for browsing, probing, and switching between checkpoints across all your experiments.

The official `tinker` CLI lists runs and checkpoints but knows nothing about
*which local experiment folder produced which run*, and has no built-in way
to verify "is this checkpoint actually serving?". `tinkpad` adds both, plus
an interactive TUI and an "active checkpoint" pointer your other scripts
can source.

## Features

- **`tinkpad ls`** — every checkpoint, every run, joined with the local
  experiment name from your `Zexp`/`Zlog/<run_id>/` folders.
- **`tinkpad probe <path>`** — fires one tiny inference request against the
  checkpoint's OpenAI-compatible endpoint; reports OK / fail / timeout +
  latency.
- **`tinkpad use <path>`** — marks one checkpoint as the active one; writes
  `~/.tinkpad/active` and a sourceable `~/.tinkpad/active.env`.
- **`tinkpad reg scan`** — walks `~/Developer` for `Zlog/<run_id>/` dirs
  and auto-registers `run_id → experiment-name`.
- **`tinkpad tui`** — Textual-based interactive browser. Two-pane layout,
  arrow keys to navigate, `p` to probe, `u` to mark active, `/` to search.

## Install

```bash
cd ~/Developer/tinkpad
python3 -m venv .venv
.venv/bin/pip install -e .[dev]
```

Then either:
```bash
.venv/bin/tinkpad doctor
```
or add to PATH:
```bash
export PATH="$HOME/Developer/tinkpad/.venv/bin:$PATH"
```

`TINKER_API_KEY` must be exported (e.g. `source ~/.local/secrets`).

## Quick start

```bash
tinkpad doctor           # confirm API reachability
tinkpad reg scan         # auto-register experiment names from ~/Developer
tinkpad runs             # list all training runs
tinkpad ls               # list all checkpoints joined with experiment names
tinkpad ls --sampler --probe   # also probe every sampler — slow but thorough
tinkpad info <tinker://…>      # detail view + probe
tinkpad probe <tinker://…>     # one-shot probe
tinkpad use   <tinker://…>     # mark active
source ~/.tinkpad/active.env   # downstream scripts inherit TINKPAD_CKPT
tinkpad tui                    # interactive browser
```

## TUI keys

| key             | action                                       |
| --------------- | -------------------------------------------- |
| arrows          | navigate runs / checkpoints                  |
| `p` / enter     | probe selected checkpoint                    |
| `a`             | probe every sampler in the current run       |
| `u`             | mark selected checkpoint as active           |
| `/`             | search experiments / runs                    |
| `r`             | refresh from API                             |
| `q`             | quit                                         |

## File layout

```
~/.tinkpad/
  registry.json     # run_id → experiment-name mapping
  scan_roots.json   # which dirs to walk for auto-scan (default ~/Developer)
  active            # current active checkpoint URI
  active.env        # sourceable: exports TINKPAD_CKPT, OPENAI_BASE_URL
  favorites.json    # (reserved)
```
