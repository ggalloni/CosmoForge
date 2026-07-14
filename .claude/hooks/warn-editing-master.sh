#!/bin/bash
# Warn when files are edited directly on master.
#
# A warning, not a block: sometimes editing master is what you want (a hotfix,
# a scratch check). But the default should be a branch, and the moment to
# notice is the first edit -- not after four files have piled up and the work
# has to be moved.

BRANCH=$(git -C "${CLAUDE_PROJECT_DIR:-.}" branch --show-current 2>/dev/null)

if [ "$BRANCH" = "master" ] || [ "$BRANCH" = "main" ]; then
  printf '%s' '{"hookSpecificOutput":{"hookEventName":"PreToolUse","additionalContext":"You are editing on '"$BRANCH"'. Unless the user asked for a direct commit, branch first (git switch -c <type>/<name>) -- uncommitted edits carry over cleanly, but only if you branch before the work piles up. Mention it to the user rather than silently continuing."}}'
fi

exit 0
