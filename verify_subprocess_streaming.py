"""Shared helpers for verify_functionality.py (repo-local, cross-platform)."""
from __future__ import annotations

import os
import shlex
import subprocess
from pathlib import Path
from typing import Mapping, Optional, Union

PathLike = Union[str, Path]


def log_message(log_path: PathLike, message: str) -> None:
    path = Path(log_path)
    with path.open("a", encoding="utf-8") as fh:
        fh.write(message.rstrip() + "\n")
    print(message, flush=True)


def log_section(log_path: PathLike, title: str) -> None:
    log_message(log_path, title)


def _cwd_for_subprocess(cwd: Optional[PathLike]) -> Optional[str]:
    if cwd is None:
        return None
    p = Path(cwd)
    # Avoid resolve() mapping a drive letter to UNC (breaks cmd.exe on Windows).
    if p.drive:
        return str(p)
    return str(p.resolve())


def stream_run(
    cmd: str,
    log_path: PathLike,
    *,
    cwd: Optional[PathLike] = None,
    env: Optional[Mapping[str, str]] = None,
    timeout: Optional[int] = None,
) -> int:
    run_cwd = _cwd_for_subprocess(cwd)
    log_message(log_path, f"$ {cmd}" + (f"  (cwd={run_cwd})" if run_cwd else ""))
    proc_env = {
        **os.environ,
        "PYTHONIOENCODING": "utf-8",
        **(env or {}),
    }
    try:
        args = shlex.split(cmd, posix=(os.name != "nt"))
        proc = subprocess.Popen(
            args,
            cwd=run_cwd,
            env=proc_env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
            shell=False,
        )
    except Exception as exc:
        log_message(log_path, f"FAILED_TO_START: {type(exc).__name__}: {exc}")
        return 1

    assert proc.stdout is not None
    try:
        for line in proc.stdout:
            log_message(log_path, line.rstrip("\n"))
        return proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        proc.kill()
        proc.wait()
        log_message(log_path, f"TIMEOUT after {timeout}s")
        return 124
