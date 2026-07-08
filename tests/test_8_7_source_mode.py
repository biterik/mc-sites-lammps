"""SPEC test 8.7: source mode (fixed-rate insertion).

Exact insertion counts per invocation; atoms appear only at
criteria-passing sites; the skip counter engages when the catalogue is
exhausted.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from util_lammps import LMP_BIN, TMPROOT, parse_dump_local, run_lammps, thermo_column

A0 = 3.52
NCELL = 2
NHOST = 4 * NCELL**3     # 32
NSITES = 4 * NCELL**3    # 32 oct sites
R_OCT = A0 / 2.0

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def write_oct_sites(path):
    basis = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]
    with open(path, "w") as f:
        for i in range(NCELL):
            for j in range(NCELL):
                for k in range(NCELL):
                    for bx, by, bz in basis:
                        f.write(f"{(i + bx) * A0:.6f} {(j + by) * A0:.6f} "
                                f"{(k + bz) * A0:.6f}\n")
    return [(round((i + b[0]) * A0, 6), round((j + b[1]) * A0, 6), round((k + b[2]) * A0, 6))
            for i in range(NCELL) for j in range(NCELL) for k in range(NCELL) for b in basis]


def source_input(rate: int, seed: int, nsteps: int, sites_arg: str = "file sites.txt",
                 extra: str = "") -> str:
    compute_line = ""
    if sites_arg == "c_S":
        compute_line = "compute S all sites/voronoi rmerge 0.3 rmin 1.6 rmax 2.0"
    return f"""
units metal
boundary p p p
atom_style atomic
lattice fcc {A0}
region box block 0 {NCELL} 0 {NCELL} 0 {NCELL}
create_box 2 box
create_atoms 1 box
mass 1 58.69
mass 2 1.008
pair_style zero 4.0
pair_coeff * *
{compute_line}
# seed {seed}
fix MC all mc/sites 1 0 2 {seed} 300.0 sites {sites_arg} mode source rate {rate}
{extra}
thermo 1
thermo_style custom step atoms f_MC[2] f_MC[8]
run {nsteps}
"""


def test_exact_insertion_counts():
    """rate insertions per invocation, exactly, until exhaustion; then the
    skip counter accounts for every missed insertion."""
    d = TMPROOT / "8_7_counts"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    write_oct_sites(d / "sites.txt")
    rate, nsteps = 5, 10
    log = run_lammps(source_input(rate, 8701, nsteps), d)
    atoms = np.array(thermo_column(log, "Atoms"), dtype=int)
    skips = np.array(thermo_column(log, "f_MC[8]"), dtype=float)
    # steps 1..6: +5 per step (30 total); step 7: +2 then exhausted (3 skips);
    # steps 8..10: +0, 5 skips each
    expected_n = [NHOST]
    expected_skip = [0.0]
    n, s = NHOST, 0.0
    for step in range(1, nsteps + 1):
        free = NHOST + NSITES - n
        ins = min(rate, free)
        n += ins
        s += rate - ins
        expected_n.append(n)
        expected_skip.append(s)
    assert np.array_equal(atoms, np.array(expected_n)), (
        f"insertion counts wrong:\n got {atoms}\n expected {expected_n}")
    assert np.array_equal(skips, np.array(expected_skip)), (
        f"skip counter wrong:\n got {skips}\n expected {expected_skip}")
    assert atoms[-1] == NHOST + NSITES


def test_atoms_appear_only_at_sites():
    """With the dynamic catalogue (rmin 1.6 -> octahedral only), every
    inserted atom must sit exactly on an octahedral site."""
    d = TMPROOT / "8_7_positions"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    oct_sites = write_oct_sites(d / "sites.txt")    # analytic positions
    extra = """
dump D all custom 1 dump.atoms id type x y z
dump_modify D format float %.10g sort id
"""
    log = run_lammps(source_input(2, 8702, 8, sites_arg="c_S", extra=extra), d)
    atoms = np.array(thermo_column(log, "Atoms"), dtype=int)
    assert atoms[-1] == NHOST + 16
    # parse last dump frame, keep species (type 2) rows
    text = (d / "dump.atoms").read_text().splitlines()
    idx = len(text) - 1 - text[::-1].index("ITEM: ATOMS id type x y z")
    nrows = atoms[-1]
    rows = np.array([list(map(float, line.split())) for line in text[idx + 1: idx + 1 + nrows]])
    species = rows[rows[:, 1] == 2]
    assert species.shape[0] == 16
    site_arr = np.array(oct_sites)
    for row in species:
        dmin = np.min(np.linalg.norm(site_arr - row[2:5], axis=1))
        assert dmin < 1e-6, f"species atom at {row[2:5]} is {dmin:.3e} from nearest oct site"
