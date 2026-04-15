#!/bin/bash
# VisOS — delegates to app.py which manages both processes via uv
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
exec uv run app.py "$@"
