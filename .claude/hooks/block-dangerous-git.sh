#!/bin/bash
# Block git commands that destroy work irrecoverably.
#
# Scope: only the commands whose damage cannot be undone. Plain `git push`,
# `git commit` and `gh pr create` are NOT blocked here -- they are recoverable,
# and already gated by `ask` permission rules.
#
# Two matching subtleties, both learned the hard way:
#
#   1. Match the WHOLE command string, not a prefix. A `permissions.deny` rule
#      matches the prefix only, so `cd /repo && git reset --hard` sails past it.
#
#   2. But a command that merely *mentions* a dangerous git command is not an
#      attempt to run it. This script blocked its own commit, because the commit
#      message documented what it blocks. So: strip heredoc bodies, and require
#      `git` to sit at a command position (start of line, or after ; && || |)
#      rather than mid-sentence.
#
# Run the self-check with:  .claude/hooks/block-dangerous-git.sh --test

DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
# shellcheck source=lib.sh
. "$DIR/lib.sh"

check() {
  local cmd="$1" reason="$2" pattern="$3"
  if runs "$cmd" "$pattern"; then
    REASON="$reason"
    return 0
  fi
  return 1
}

# Returns 0 (dangerous) or 1 (safe). Sets REASON.
is_dangerous() {
  local cmd="$1"
  REASON=""

  check "$cmd" "git checkout -- <file> discards uncommitted changes to that file." \
    'git[[:space:]]+checkout[[:space:]]+--[[:space:]]' && return 0

  check "$cmd" "git checkout . discards every uncommitted change in the tree." \
    'git[[:space:]]+checkout[[:space:]]+\.([[:space:]]|$)' && return 0

  # git restore: --staged alone only unstages (safe). Anything touching the
  # worktree discards changes.
  if check "$cmd" "" 'git[[:space:]]+restore([[:space:]]|$)'; then
    if has_flag "$cmd" '--staged' && ! has_flag "$cmd" '--worktree'; then
      : # `git restore --staged <file>` unstages without touching the worktree.
    else
      REASON="git restore <file> discards uncommitted changes. Only --staged (unstage) is allowed."
      return 0
    fi
  fi

  check "$cmd" "git reset --hard discards all uncommitted work in the tree." \
    'git[[:space:]]+reset[[:space:]].*--hard' && return 0

  check "$cmd" "git clean -f deletes untracked files, which exist in no commit." \
    'git[[:space:]]+clean[[:space:]]+-[a-zA-Z]*f' && return 0

  check "$cmd" "git branch -D force-deletes a branch, including unmerged commits. Use -d." \
    'git[[:space:]]+branch[[:space:]]+-D' && return 0

  # Force push. The flag can precede or follow the refspec, so check it
  # separately from the `git push` match rather than in one anchored regex.
  if check "$cmd" "" 'git[[:space:]]+push([[:space:]]|$)'; then
    if has_flag "$cmd" '(^|[[:space:]])(--force-with-lease|--force|-f)([[:space:]]|=|$)'; then
      REASON="A force push rewrites published history and can destroy others' commits."
      return 0
    fi
  fi

  return 1
}

# ---------------------------------------------------------------- self-check
if [ "$1" = "--test" ]; then
  fail=0
  expect() {  # expect BLOCK|PASS <command>
    if is_dangerous "$2"; then got=BLOCK; else got=PASS; fi
    if [ "$got" = "$1" ]; then printf '  ok   %-6s %s\n' "$got" "$2"
    else printf '  FAIL expected %s, got %s: %s\n' "$1" "$got" "$2"; fail=1; fi
  }

  expect BLOCK 'git checkout -- src/foo.py'
  expect BLOCK 'cd /repo && git checkout -- src/foo.py'
  expect BLOCK 'git checkout .'
  expect BLOCK 'git restore src/foo.py'
  expect BLOCK 'git restore --staged --worktree foo.py'
  expect BLOCK 'git reset --hard'
  expect BLOCK 'git reset --hard origin/master'
  expect BLOCK 'git clean -fd'
  expect BLOCK 'git clean -fdx'
  expect BLOCK 'git branch -D feature/x'
  expect BLOCK 'git push -f'
  expect BLOCK 'git push --force origin master'
  expect BLOCK 'git push origin master --force'
  expect BLOCK 'git push --force-with-lease'

  expect PASS 'git push'
  expect PASS 'git push origin my-branch'
  expect PASS 'git push -u origin docs/foo'
  expect PASS 'git checkout -b feature/x'
  expect PASS 'git switch -c feature/x'
  expect PASS 'git checkout master'
  expect PASS 'git restore --staged foo.py'
  expect PASS 'git commit -m "x"'
  expect PASS 'git branch -d merged'
  expect PASS 'git reset HEAD~1'
  expect PASS 'git status --short'
  expect PASS 'git add -f ignored.txt'
  expect PASS 'uv run pytest -q'
  expect PASS 'grep -f patterns.txt file.txt'

  # The regression that blocked this script's own commit: prose that mentions
  # a dangerous command is not an attempt to run it.
  expect PASS 'git commit -m "docs: never run git reset --hard on master"'
  expect PASS "$(printf 'git commit -F - <<EOF\nchore: block git checkout -- <file>\n\nAlso blocks: cd /r && git reset --hard\nEOF')"
  expect PASS "$(printf 'cat <<'"'"'EOF'"'"' > notes.md\nAvoid git clean -fd here.\nEOF')"

  echo
  [ $fail -eq 0 ] && echo "ALL GREEN" || echo "FAILURES"
  exit $fail
fi

# ------------------------------------------------------------------ the hook
INPUT=$(cat)
COMMAND=$(printf '%s' "$INPUT" | jq -r '.tool_input.command')

if is_dangerous "$COMMAND"; then
  {
    echo "BLOCKED: $REASON"
    echo ""
    echo "This destroys work irrecoverably, and you do not have authority to run it."
    echo "Ask the user; let them run it themselves if they confirm."
  } >&2
  exit 2
fi

exit 0
