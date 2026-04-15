#!/usr/bin/env python3
"""
VisOS launcher.

Run with `uv run app.py` from the project root. Starts:
  • FastAPI backend  → http://localhost:8000
  • Next.js frontend → http://localhost:3000

Does NOT open a browser. Prints the frontend URL once both servers are ready.
Ctrl-C cleanly stops both processes.
"""

from __future__ import annotations

import os
import signal
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).parent.resolve()
BACKEND_DIR = ROOT / "backend"
LOG_DIR = ROOT / ".logs"
LOG_DIR.mkdir(exist_ok=True)
BACK_LOG = LOG_DIR / "backend.log"
FRONT_LOG = LOG_DIR / "frontend.log"

IS_WIN = sys.platform == "win32"

# ── Terminal colours ─────────────────────────────────────────────────────────
G = "\033[92m"
Y = "\033[93m"
R = "\033[91m"
C = "\033[96m"
B = "\033[94m"
W = "\033[0m"
BOLD = "\033[1m"


def ok(msg: str) -> None:
    print(f"  {G}✓{W}  {msg}")


def info(msg: str) -> None:
    print(f"  {C}→{W}  {msg}")


def warn(msg: str) -> None:
    print(f"  {Y}!{W}  {msg}")


def err(msg: str) -> None:
    print(f"  {R}✗{W}  {msg}")


# ── Health checks ────────────────────────────────────────────────────────────
def wait_for(url: str, timeout: int = 120) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            urllib.request.urlopen(url, timeout=1)
            return True
        except (urllib.error.URLError, urllib.error.HTTPError, ConnectionError, OSError):
            time.sleep(0.5)
    return False


# ── Stream forwarder ─────────────────────────────────────────────────────────
def _tee(stream, log_file, prefix: str, colour: str, stop: threading.Event) -> None:
    for raw in iter(stream.readline, b""):
        if stop.is_set():
            break
        try:
            text = raw.decode("utf-8", errors="replace")
            log_file.write(text)
            log_file.flush()
        except Exception:
            pass
        try:
            sys.stdout.write(f"{colour}[{prefix}]{W} {text}")
            sys.stdout.flush()
        except Exception:
            pass


# ── Process launchers ────────────────────────────────────────────────────────
def start_backend() -> subprocess.Popen:
    """Run the FastAPI server via uvicorn inside the current uv environment."""
    log = open(BACK_LOG, "ab", buffering=0)
    log.write(
        f"\n\n{'='*60}\n  BACKEND START  {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n".encode()
    )

    cmd = [
        sys.executable, "-m", "uvicorn", "main:app",
        "--reload",
        "--host", "0.0.0.0",
        "--port", "8000",
        "--log-level", "info",
    ]
    kwargs = {
        "cwd": BACKEND_DIR,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
    }
    if IS_WIN:
        proc = subprocess.Popen(cmd, **kwargs, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(cmd, **kwargs, start_new_session=True)

    stop = threading.Event()
    threading.Thread(
        target=_tee, args=(proc.stdout, log, "BACK ", C, stop), daemon=True
    ).start()
    proc._tee_stop = stop  # type: ignore[attr-defined]
    return proc


def start_frontend() -> subprocess.Popen:
    """Run the Next.js dev server on port 3000."""
    log = open(FRONT_LOG, "ab", buffering=0)
    log.write(
        f"\n\n{'='*60}\n  FRONTEND START  {time.strftime('%Y-%m-%d %H:%M:%S')}\n{'='*60}\n".encode()
    )

    pm = "pnpm" if _has("pnpm") else "npm"
    if not (ROOT / "node_modules").exists():
        info(f"Installing Node.js dependencies ({pm} install)…")
        install = subprocess.run(f"{pm} install", cwd=ROOT, shell=True)
        if install.returncode != 0:
            err("Node dependency install failed — aborting.")
            sys.exit(1)

    cmd = f"{pm} run dev -- -p 3000"
    kwargs = {
        "cwd": ROOT,
        "stdout": subprocess.PIPE,
        "stderr": subprocess.STDOUT,
        "shell": True,
    }
    if IS_WIN:
        proc = subprocess.Popen(cmd, **kwargs, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
    else:
        proc = subprocess.Popen(cmd, **kwargs, start_new_session=True)

    stop = threading.Event()
    threading.Thread(
        target=_tee, args=(proc.stdout, log, "FRONT", B, stop), daemon=True
    ).start()
    proc._tee_stop = stop  # type: ignore[attr-defined]
    return proc


def _has(cmd: str) -> bool:
    try:
        subprocess.run([cmd, "--version"], capture_output=True, check=True)
        return True
    except Exception:
        return False


# ── Shutdown ─────────────────────────────────────────────────────────────────
def kill_proc(proc: subprocess.Popen, name: str) -> None:
    if proc.poll() is not None:
        return
    info(f"Stopping {name} (PID {proc.pid})…")
    try:
        if IS_WIN:
            subprocess.run(["taskkill", "/F", "/T", "/PID", str(proc.pid)], capture_output=True)
        else:
            pgid = os.getpgid(proc.pid)
            os.killpg(pgid, signal.SIGTERM)
            for _ in range(50):
                time.sleep(0.1)
                if proc.poll() is not None:
                    return
            os.killpg(pgid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    except Exception as exc:
        warn(f"Could not stop {name}: {exc}")


def main() -> int:
    print(f"\n{BOLD}{C}VisOS — launching…{W}\n")

    backend = start_backend()
    frontend = start_frontend()

    def _shutdown(*_args: object) -> None:
        print()
        info("Shutting down…")
        kill_proc(frontend, "frontend")
        kill_proc(backend, "backend")
        sys.exit(0)

    signal.signal(signal.SIGINT, _shutdown)
    signal.signal(signal.SIGTERM, _shutdown)

    backend_up = wait_for("http://localhost:8000/api/health", timeout=120)
    if backend_up:
        ok("Backend ready  → http://localhost:8000")
    else:
        warn("Backend did not respond within 120s — check .logs/backend.log")

    frontend_up = wait_for("http://localhost:3000", timeout=180)
    if frontend_up:
        ok("Frontend ready → http://localhost:3000")
    else:
        warn("Frontend did not respond within 180s — check .logs/frontend.log")

    print()
    print(f"  {BOLD}{G}WebUI launched at http://localhost:3000{W}")
    print(f"  Backend API at http://localhost:8000")
    print(f"  Ctrl-C to stop.\n")

    try:
        while True:
            if backend.poll() is not None:
                err("Backend exited unexpectedly.")
                kill_proc(frontend, "frontend")
                return 1
            if frontend.poll() is not None:
                err("Frontend exited unexpectedly.")
                kill_proc(backend, "backend")
                return 1
            time.sleep(1)
    except KeyboardInterrupt:
        _shutdown()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
