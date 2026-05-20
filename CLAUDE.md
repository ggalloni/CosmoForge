## Agent skills

### Issue tracker

GitHub Issues on `ggalloni/CosmoForge`, accessed via the `gh` CLI. See `docs/agents/issue-tracker.md`.

### Triage labels

Canonical label vocabulary (`needs-triage`, `needs-info`, `ready-for-agent`, `ready-for-human`, `wontfix`) — used as-is. See `docs/agents/triage-labels.md`.

### Domain docs

Single-context: one `CONTEXT.md` at repo root, ADRs under `docs/adr/`. See `docs/agents/domain.md`.

## Documentation drift

At the end of every feature implementation that touches public surface — renamed/added/removed modules, changed signatures or kwargs, new or removed behaviour — do a quick doc-drift sweep before declaring the work done:

1. `grep -rn '<old-name>' docs/ CONTEXT.md README.md src/cosmoforge.*/README.md` for every symbol or path that was renamed or removed.
2. `uv run --group docs sphinx-build -b html docs/source docs/build/html` and confirm no *new* warnings beyond the pre-existing docstring noise.
3. Skim CONTEXT.md and the per-package READMEs for sections that describe the touched area; update or note as stale.

Skip for purely internal refactors that change no public surface.
