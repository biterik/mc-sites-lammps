"""Shared helpers for the MC-SITES-LAMMPS validation suite.

Runs the locally built lmp binary on generated inputs and parses
`dump local` / `fix print` / thermo output. Every test that uses
randomness records its seed in the input file it writes.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import os
import subprocess
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LMP_BIN = Path(os.environ.get("LMP_BIN", REPO / "lammps" / "build" / "lmp"))
MPIRUN = os.environ.get("MPIRUN", "mpirun")
TMPROOT = REPO / "tests" / "tmp"


def run_lammps(input_text: str, workdir: Path, nprocs: int = 1,
               name: str = "in.test") -> str:
    """Run lmp on input_text inside workdir; return the log text.

    Raises RuntimeError with the log tail on nonzero exit.
    """
    workdir.mkdir(parents=True, exist_ok=True)
    infile = workdir / name
    infile.write_text(input_text)
    logfile = workdir / "log.lammps"
    if nprocs == 1:
        cmd = [str(LMP_BIN)]
    else:
        cmd = [MPIRUN, "-np", str(nprocs), str(LMP_BIN)]
    cmd += ["-in", str(infile), "-log", str(logfile), "-screen", "none", "-nocite"]
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True)
    log = logfile.read_text() if logfile.exists() else ""
    if proc.returncode != 0:
        raise RuntimeError(
            f"LAMMPS failed (rc={proc.returncode}) in {workdir}\n"
            f"--- stderr ---\n{proc.stderr[-2000:]}\n--- log tail ---\n{log[-3000:]}"
        )
    return log


def parse_dump_local(path: Path) -> dict[int, np.ndarray]:
    """Parse a `dump local` file into {timestep: array(nrows, ncols)}."""
    frames: dict[int, np.ndarray] = {}
    lines = Path(path).read_text().splitlines()
    i = 0
    while i < len(lines):
        if lines[i].startswith("ITEM: TIMESTEP"):
            step = int(lines[i + 1])
            assert lines[i + 2].startswith("ITEM: NUMBER OF ENTRIES")
            n = int(lines[i + 3])
            j = i + 4
            # skip box bounds + entries header
            while not lines[j].startswith("ITEM: ENTRIES"):
                j += 1
            rows = [list(map(float, lines[j + 1 + k].split())) for k in range(n)]
            frames[step] = np.array(rows).reshape(n, -1) if n else np.zeros((0, 0))
            i = j + 1 + n
        else:
            i += 1
    return frames


def thermo_column(log: str, col: str) -> list[float]:
    """Extract a thermo column (all run sections concatenated)."""
    vals: list[float] = []
    lines = log.splitlines()
    for i, line in enumerate(lines):
        toks = line.split()
        if col in toks and toks[0] in ("Step", "step"):
            idx = toks.index(col)
            j = i + 1
            while j < len(lines):
                if lines[j].startswith("WARNING"):    # warnings interleave with rows
                    j += 1
                    continue
                t2 = lines[j].split()
                if not t2:
                    break
                try:
                    int(t2[0])    # data rows start with the integer step
                    vals.append(float(t2[idx]))
                except (ValueError, IndexError):
                    break
                j += 1
    return vals
