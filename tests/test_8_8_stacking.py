"""SPEC test 8.8: multi-species stacking.

Two fix mc/sites instances (species types 2 and 3, disjoint site lists:
octahedral vs tetrahedral) coexist; each reproduces its own Langmuir
point and its own counters. The duplicate-species-type error path must
trigger.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from util_lammps import LMP_BIN, TMPROOT, run_lammps, thermo_column

KB = 8.617333262e-5
A0 = 3.52
NCELL = 3
NHOST = 4 * NCELL**3     # 108
N_OCT = 4 * NCELL**3     # 108
N_TET = 8 * NCELL**3     # 216

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def write_site_files(d):
    oct_basis = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]
    tet_basis = [(0.25, 0.25, 0.25), (0.75, 0.25, 0.25), (0.25, 0.75, 0.25),
                 (0.25, 0.25, 0.75), (0.75, 0.75, 0.25), (0.75, 0.25, 0.75),
                 (0.25, 0.75, 0.75), (0.75, 0.75, 0.75)]
    for name, basis in (("oct.txt", oct_basis), ("tet.txt", tet_basis)):
        with open(d / name, "w") as f:
            for i in range(NCELL):
                for j in range(NCELL):
                    for k in range(NCELL):
                        for bx, by, bz in basis:
                            f.write(f"{(i + bx) * A0:.6f} {(j + by) * A0:.6f} "
                                    f"{(k + bz) * A0:.6f}\n")


BASE = f"""
units metal
boundary p p p
atom_style atomic
lattice fcc {A0}
region box block 0 {NCELL} 0 {NCELL} 0 {NCELL}
create_box 3 box
create_atoms 1 box
mass 1 58.69
mass 2 1.008
mass 3 4.0026
pair_style zero 4.0
pair_coeff * *
"""


def theta_exact(mu, T):
    return 1.0 / (1.0 + math.exp(-mu / (KB * T)))


def test_two_species_coexist():
    d = TMPROOT / "8_8_stacking"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    write_site_files(d)
    mu2, mu3, T = +0.02, -0.03, 300.0
    n_equil, n_sample = 300, 1200
    log = run_lammps(BASE + f"""
# seeds 8801 / 8802
fix MCA all mc/sites 1 40 2 8801 {T} sites file oct.txt mode gc mu {mu2}
fix MCB all mc/sites 1 60 3 8802 {T} sites file tet.txt mode gc mu {mu3}
thermo 1
thermo_style custom step atoms f_MCA[4] f_MCA[5] f_MCB[4] f_MCB[5]
run {n_equil}
run {n_sample}
""", d)
    n2 = np.array(thermo_column(log, "f_MCA[4]")[-n_sample:])
    m2 = np.array(thermo_column(log, "f_MCA[5]")[-n_sample:])
    n3 = np.array(thermo_column(log, "f_MCB[4]")[-n_sample:])
    m3 = np.array(thermo_column(log, "f_MCB[5]")[-n_sample:])
    atoms = np.array(thermo_column(log, "Atoms")[-n_sample:])

    # per-species catalogue sizes and cross-consistent totals
    assert np.all(m2 == N_OCT), f"species-2 catalogue: {np.unique(m2)}"
    assert np.all(m3 == N_TET), f"species-3 catalogue: {np.unique(m3)}"
    assert np.array_equal(atoms, NHOST + n2 + n3), "per-species counters inconsistent"

    th2, th3 = float(n2.mean()) / N_OCT, float(n3.mean()) / N_TET
    ex2, ex3 = theta_exact(mu2, T), theta_exact(mu3, T)
    assert abs(th2 - ex2) < 0.02, f"species 2: {th2:.4f} vs {ex2:.4f}"
    assert abs(th3 - ex3) < 0.02, f"species 3: {th3:.4f} vs {ex3:.4f}"
    print(f"stacked isotherms: type2 {th2:.4f}/{ex2:.4f}, type3 {th3:.4f}/{ex3:.4f}")


def test_duplicate_species_type_errors():
    d = TMPROOT / "8_8_duplicate"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    write_site_files(d)
    with pytest.raises(RuntimeError) as exc:
        run_lammps(BASE + """
fix MCA all mc/sites 1 10 2 8803 300.0 sites file oct.txt mode gc mu 0.0
fix MCB all mc/sites 1 10 2 8804 300.0 sites file tet.txt mode gc mu 0.0
run 1
""", d)
    assert "both drive atom type" in str(exc.value), (
        f"expected the duplicate-species error, got:\n{str(exc.value)[-500:]}")
