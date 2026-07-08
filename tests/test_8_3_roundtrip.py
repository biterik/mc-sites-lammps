"""SPEC test 8.3: round-trip guard (fix mc/sites, pair zero + lj/cut).

Insert-then-delete every site must return total energy to baseline within
1e-9 eV and N to baseline; the reject paths must leave the state unchanged
(energy re-check). Deterministic driving via extreme mu (equal-style
variable switched between runs): mu=+10 accepts every insertion and
rejects every deletion; mu=-10 the reverse. Seeds recorded in inputs.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from util_lammps import LMP_BIN, TMPROOT, run_lammps, thermo_column

A0 = 3.52
NCELL = 2
NHOST = 4 * NCELL**3      # 32
NSITES = 4 * NCELL**3     # 32 octahedral sites

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def write_oct_sites(path):
    """All octahedral sites of the NCELL^3 fcc box (4 per cell)."""
    basis = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]
    with open(path, "w") as f:
        f.write("# fcc octahedral sites\n")
        for i in range(NCELL):
            for j in range(NCELL):
                for k in range(NCELL):
                    for bx, by, bz in basis:
                        f.write(f"{(i + bx) * A0:.6f} {(j + by) * A0:.6f} "
                                f"{(k + bz) * A0:.6f}\n")


def base_input(pair_block: str, seed: int, mu_schedule: str) -> str:
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
{pair_block}
variable mu equal 10.0
# seed {seed}
fix MC all mc/sites 1 200 2 {seed} 300.0 sites file sites.txt mode gc mu v_mu
thermo 1
thermo_style custom step atoms pe f_MC[1] f_MC[2] f_MC[3]
thermo_modify format float %.15g
run 0
{mu_schedule}
"""

PAIR_ZERO = """
pair_style zero 4.0
pair_coeff * *
"""

PAIR_LJ = """
pair_style lj/cut 5.0
pair_coeff 1 1 0.10 2.20
pair_coeff 1 2 0.05 1.50
pair_coeff 2 2 0.05 1.00
"""

FILL_EMPTY_SCHEDULE = """
# fill: mu=+10 accepts every insertion, rejects every deletion
run 5
# empty: mu=-10 accepts every deletion, rejects every insertion
variable mu equal -10.0
run 5
# reject-only phase on the empty lattice: nothing may change
run 3
"""


def run_case(name: str, pair_block: str, seed: int = 4242):
    d = TMPROOT / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    write_oct_sites(d / "sites.txt")
    log = run_lammps(base_input(pair_block, seed, FILL_EMPTY_SCHEDULE), d)
    atoms = thermo_column(log, "Atoms")
    pe = thermo_column(log, "PotEng")
    return np.array(atoms), np.array(pe)


@pytest.mark.parametrize("pair_name,pair_block", [("zero", PAIR_ZERO), ("lj", PAIR_LJ)])
def test_fill_then_empty_returns_baseline(pair_name, pair_block):
    atoms, pe = run_case(f"8_3_cycle_{pair_name}", pair_block)
    n0, e0 = atoms[0], pe[0]
    assert n0 == NHOST
    # after the fill phase every site is occupied
    assert atoms[6] == NHOST + NSITES, (
        f"{pair_name}: fill phase reached N={atoms[6]}, expected {NHOST + NSITES}")
    # fill sticks: deletions all rejected at mu=+10
    assert np.all(atoms[3:7] <= NHOST + NSITES)
    # after the empty phase N and E return to baseline
    assert atoms[11] == n0, f"{pair_name}: N={atoms[11]} after empty phase, expected {n0}"
    assert abs(pe[11] - e0) < 1e-9, (
        f"{pair_name}: energy after fill+empty cycle differs from baseline by "
        f"{abs(pe[11] - e0):.3e} eV")
    # reject-only phase on the empty lattice: N and E frozen
    assert np.all(atoms[11:] == n0)
    assert np.max(np.abs(pe[11:] - e0)) < 1e-9, (
        f"{pair_name}: reject path changed the energy by "
        f"{np.max(np.abs(pe[11:] - e0)):.3e} eV")
    print(f"{pair_name}: baseline E={e0:.12g}, fill N={atoms[6]:.0f}, "
          f"cycle dE={abs(pe[11] - e0):.2e}")


def test_deletion_reject_leaves_state_unchanged():
    """Filled lattice + mu=+10: every proposal on an occupied site is a
    deletion trial that must be rejected and exactly restored."""
    d = TMPROOT / "8_3_delete_reject"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    write_oct_sites(d / "sites.txt")
    schedule = """
run 5
# lattice now full; keep mu=+10: all deletion trials must be rejected
run 5
"""
    log = run_lammps(base_input(PAIR_LJ, 777, schedule), d)
    atoms = np.array(thermo_column(log, "Atoms"))
    pe = np.array(thermo_column(log, "PotEng"))
    nfull = NHOST + NSITES
    assert atoms[6] == nfull
    e_full = pe[6]
    assert np.all(atoms[6:] == nfull), "deletion-reject changed N"
    assert np.max(np.abs(pe[6:] - e_full)) < 1e-9, (
        f"deletion-reject changed E by {np.max(np.abs(pe[6:] - e_full)):.3e} eV "
        "(pack/unpack_exchange restore broken)")


def test_seed_reproducibility():
    """Identical input + seed => identical trajectory (N per step)."""
    runs = []
    for rep in (1, 2):
        d = TMPROOT / f"8_3_repro_{rep}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        write_oct_sites(d / "sites.txt")
        inp = base_input(PAIR_LJ, 999, "variable mu equal 0.0\nrun 20")
        log = run_lammps(inp, d)
        runs.append((np.array(thermo_column(log, "Atoms")),
                     np.array(thermo_column(log, "PotEng"))))
    assert np.array_equal(runs[0][0], runs[1][0]), "N trajectory not reproducible"
    assert np.array_equal(runs[0][1], runs[1][1]), "E trajectory not reproducible"
