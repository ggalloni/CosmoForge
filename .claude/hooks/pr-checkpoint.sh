#!/bin/bash
# Remind to run the over-engineering review when a PR is being opened or merged.
#
# The review is already the routed reviewer, but it gets skipped exactly when it
# matters -- while wrapping up. This fires at the two commands that mean "this
# PR is done". A nudge, not a block: pushing work-in-progress is untouched.
#
# Uses lib.sh so that a commit message *mentioning* `gh pr create` does not
# trigger it. The first version did, on its own commit.
#
# Self-check:  .claude/hooks/pr-checkpoint.sh --test

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$DIR/lib.sh"

is_pr_completion() {
  runs "$1" 'gh[[:space:]]+pr[[:space:]]+(create|merge)([[:space:]]|$)'
}

if [ "$1" = "--test" ]; then
  fail=0
  expect() {
    if is_pr_completion "$2"; then got=FIRE; else got=QUIET; fi
    if [ "$got" = "$1" ]; then printf '  ok   %-5s %s\n' "$got" "$2"
    else printf '  FAIL expected %s, got %s: %s\n' "$1" "$got" "$2"; fail=1; fi
  }

  expect FIRE  'gh pr create --base master'
  expect FIRE  'cd /repo && gh pr create --fill'
  expect FIRE  'gh pr merge 45 --squash'

  expect QUIET 'gh pr list'
  expect QUIET 'gh pr view 45 --json reviews'
  expect QUIET 'gh pr checks 45'
  expect QUIET 'git status'
  # The regression: prose that mentions the command is not the command.
  expect QUIET 'git commit -m "chore: remind on gh pr create"'
  expect QUIET "$(printf 'git commit -F - <<MSG\nchore: fires on `gh pr create` / `gh pr merge`\nMSG')"

  echo
  [ $fail -eq 0 ] && echo "ALL GREEN" || echo "FAILURES"
  exit $fail
fi

COMMAND=$(jq -r '.tool_input.command')

if is_pr_completion "$COMMAND"; then
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"PR checkpoint: unless you already ran it on the current diff this session, run the ponytail-review skill on the full PR diff before this command completes. It hunts over-engineering, which CI and Copilot do not. Report what it finds; do not skip silently."}}'
fi

exit 0
