#!/bin/bash
# Shared matching logic for the Bash PreToolUse hooks.
#
# The problem both hooks have: a command that *mentions* `git reset --hard` or
# `gh pr create` -- in a commit message, a heredoc, a doc string -- is not an
# attempt to run it. Grepping the raw command string cries wolf on exactly the
# commits that document what the hooks do. (Both hooks misfired on their own
# commit before this existed.)
#
# So: strip heredoc bodies, then require the match to sit at a command position.

# Remove heredoc bodies (`<<EOF ... EOF`, `<<'MSG' ... MSG`), keeping the line
# that opens them.
strip_heredocs() {
  awk '
    !inside && match($0, /<<-?[[:space:]]*'\''?[A-Za-z_][A-Za-z0-9_]*'\''?/) {
      d = substr($0, RSTART, RLENGTH)
      sub(/^<<-?[[:space:]]*'\''?/, "", d)
      sub(/'\''$/, "", d)
      delim = d; inside = 1; print; next
    }
    inside && $0 ~ "^[[:space:]]*" delim "[[:space:]]*$" { inside = 0; next }
    inside { next }
    { print }
  '
}

# A command position: the start of a line, or just after a shell separator.
# Prose mentions ("run `git reset --hard`") sit mid-line and so do not match.
#
# [;&|] covers `&&` and `||` for free -- the second `&` / `|` is itself a match,
# with the separator's own characters consumed by the class. No extra alternative
# is needed, and one here would be dead regex.
#
# Known limitation: a dangerous command quoted inside another command
# ("echo 'git reset --hard'") still sits at a command position and will match.
# Heredocs are stripped; general quote awareness would mean parsing shell in
# regex. The failure direction is safe -- it blocks, and you run it yourself.
_AT_CMD='(^|[;&|])[[:space:]]*'

# runs <command-string> <extended-regex>
# True when the command actually invokes something matching the regex.
runs() {
  printf '%s\n' "$1" | strip_heredocs | grep -qE "$_AT_CMD$2"
}

# has_flag <command-string> <extended-regex-of-flag>
# True when the flag appears anywhere in the real command (not a heredoc body).
# Use only after `runs` has confirmed the subcommand, since a bare flag match
# is not anchored.
# The `--` matters: the patterns here start with `--`, which grep would
# otherwise read as its own option.
has_flag() {
  printf '%s\n' "$1" | strip_heredocs | grep -qE -- "$2"
}
