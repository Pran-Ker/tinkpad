# TODO — drafts and follow-ups

## Filed

- [#44 — `tinker checkpoint probe <path>` — verify a sampler actually serves](https://github.com/thinking-machines-lab/tinker/issues/44)

## Issues still to file on `thinking-machines-lab/tinker`

For each: file the body first, then drop the publicity comment as a follow-up.

---

### Friendly run names in `tinker run list` / `checkpoint list`

**Body:**

Every run shows up as a UUID. That's fine when you have five of them, but I'm at 74 now, all on the same base model, and `tinker run list` has stopped being scannable. The only thing distinguishing rows is `last_request_time`, which only helps if you remember when you launched what. After a couple of weeks you don't.

What would actually solve this is a free-form name attached to a run at creation time, or settable later via something like `tinker run set-name <id> <label>`. Either as a first-class field on `TrainingRun`, or surfaced out of the `user_metadata` blob that already exists on the type and printed by default in `run list` and `checkpoint list`. The CLI already has the column space for it.

Native support would mean labels show up wherever the CLI runs, including over SSH or on a teammate's box.

**Follow-up comment:**

Built a workaround while waiting. A local registry that maps run IDs to folder names, plus a TUI for browsing.

https://github.com/Pran-Ker/tinkpad

---

### `list_training_runs` paginates silently at limit=20

**Body:**

Default limit is 20. With 74 runs in my account, runs 21+ silently disappear from `list_training_runs()` output. No warning, no hint to bump `--limit`. Lost an hour the first time chasing runs I thought were deleted.

Two options:
1. CLI defaults to paginate transparently to `total_count`, with `--limit` to cap.
2. Or at minimum, print `(showing 20 of 74)` in the footer when there's more.

Same applies to `list_user_checkpoints`.

**Follow-up comment:**

Worked around this in tinkpad by walking the cursor manually until `total_count`. Native fix would let me drop that.

https://github.com/Pran-Ker/tinkpad
