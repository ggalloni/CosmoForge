## What changed

<!-- One paragraph. What does this do, and why? -->

## Public surface

<!--
List anything a user touches: new/renamed/removed kwargs, changed defaults,
new or removed config keys, new behaviour. Write "none" for a pure internal
refactor.
-->

## Checklist

- [ ] `docs/source/changelog.rst` updated under **Unreleased** — or the
      `skip-changelog` label applied because this PR has no user-visible surface.
- [ ] ADR added or amended if the decision is hard to reverse and surprising
      without context (`docs/adr/`, see `INDEX.md` for the protocol).
- [ ] `CONTEXT.md` updated if this introduces or sharpens a domain term.
- [ ] Doc-drift sweep done if public surface moved (see `CLAUDE.md`).
- [ ] Tests pass per package (`uv run pytest src/cosmoforge.<pkg>/tests/ -s`).
