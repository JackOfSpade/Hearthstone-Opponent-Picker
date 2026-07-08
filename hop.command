#!/bin/zsh
# hop — double-click launcher for the Hearthstone Opponent Picker menu-bar app.
#
# Unlike an Electron app (see ~/Desktop/Infinite Canvas.command), hop is pure Python
# installed EDITABLE (`pip install -e .`), so its source is ALWAYS live — there is
# nothing to build and no staleness window for code changes. The only thing an
# editable install does not pick up automatically is a change to pyproject.toml (a new
# dependency, a changed entry point), because package METADATA is frozen at install
# time. So this refreshes the install only when pyproject.toml is newer than the
# install marker; every other launch is instant.
#
# Double-click this in Finder. A Terminal window opens and stays open while the app
# runs (it carries the logs); quit the menu-bar app or close the window to stop.

# This script's own directory IS the project root. `:A` resolves symlinks, so a copy
# aliased onto the Desktop still finds the real tree.
PROJECT_DIR="${0:A:h}"
cd "$PROJECT_DIR" || { echo "ERROR: project dir not found: $PROJECT_DIR"; exit 1; }

notify() { osascript -e "display notification \"$1\" with title \"hop\" sound name \"$2\"" >/dev/null 2>&1; }

# hop is installed into ONE python (the framework 3.14 that has numpy/Pillow/pyobjc).
# `python3` on the login PATH may be a different one (e.g. homebrew) that cannot import
# hop, so pick the first interpreter that actually can — never just the first python3.
pick_python() {
  for cand in \
      python3 \
      /Library/Frameworks/Python.framework/Versions/Current/bin/python3 \
      /Library/Frameworks/Python.framework/Versions/3.14/bin/python3 \
      /usr/local/bin/python3 /opt/homebrew/bin/python3; do
    command -v "$cand" >/dev/null 2>&1 || continue
    if "$cand" -c 'import hop' >/dev/null 2>&1; then print -r -- "$cand"; return 0; fi
  done
  return 1
}

PY="$(pick_python)" || {
  echo "x No python3 can import hop. Install it once with:"
  echo "    pip install -e '$PROJECT_DIR'"
  notify "hop is not installed. See the Terminal window." "Basso"
  exit 1
}
echo "> Using $PY"

# Refresh the editable install only when project metadata changed (deps/entry points).
# Source-code edits need no refresh — they are already live under `pip install -e .`.
MARKER="hop.egg-info/PKG-INFO"
if [ ! -e "$MARKER" ] || [ pyproject.toml -nt "$MARKER" ]; then
  echo "> pyproject.toml changed since the last install — refreshing dependencies..."
  if "$PY" -m pip install -e . --quiet; then
    notify "hop dependencies refreshed." "Glass"
  else
    echo "x pip install failed — launching anyway on the existing install."
    notify "Dependency refresh failed; using existing install." "Basso"
  fi
else
  echo "> Up to date — launching."
fi

# The menu-bar control panel needs pyobjc; without it, fall back to the web dashboard
# (localhost) so a double-click still gives a usable surface.
if "$PY" -c 'import AppKit' >/dev/null 2>&1; then
  echo "> Launching the menu-bar app — look for the target icon in your menu bar."
  echo "  (This Terminal window stays open while it runs; quit the app to stop.)"
  exec "$PY" -m hop.cli app
else
  echo "> pyobjc is not installed, so the menu bar is unavailable."
  echo "  Opening the web dashboard instead (pip install 'hop[mac]' for the menu bar)."
  exec "$PY" -m hop.cli dashboard
fi
