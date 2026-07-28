"""Test 8.10: high-rank regression (viper 8B follow-up, 2026-07-28).

The 43-test suite exercised np <= 4; the viper 8B session raised (and the
2026-07-28 container investigation could not reproduce) a rank-count
acceptance discrepancy at np >> 4.  This test closes the coverage gap:
an oversubscribed np=16 hybrid MD/MC run -- subdomains hold ~16 host
atoms, comparable to the 128-rank production regime -- with the fix's
`check yes` state-consistency verification active (atom count, species
accounting, Metropolis reference energy, every block), plus a loose
statistical acceptance band against the same run at np=1.

Runs oversubscribed on any core count (OMPI_MCA_rmaps_base_oversubscribe
or slurm equivalents); skipped when mpirun is unavailable.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from util_lammps import LMP_BIN, MPIRUN, TMPROOT, run_lammps, thermo_column

A0 = 3.52
NCELL = 4          # 256 host atoms; np=16 -> ~16 atoms/rank
T = 300.0
MU = -0.02         # lj/cut species-host regime with mid-band acceptance
SEED = 8101
NEQUIL = 100
NMD = 400
NEVERY = 10        # 40 blocks x NTRIALS trials
NTRIALS = 25

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def hybrid_input() -> str:
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
pair_style lj/cut 5.0
pair_coeff 1 1 0.05 2.2
pair_coeff 1 2 0.02 1.8
pair_coeff 2 2 0.005 1.5
compute S all sites/voronoi rmerge 0.3 rmin 1.6 rmax 2.0
velocity all create 30.0 {SEED} mom yes rot yes
fix NVT all nvt temp 30.0 30.0 0.1
timestep 0.001
thermo 50
run {NEQUIL}
# seed {SEED}
fix MC all mc/sites {NEVERY} {NTRIALS} 2 {SEED} {T} sites c_S mode gc mu {MU} check yes
thermo_style custom step atoms pe f_MC[1] f_MC[4] f_MC[5] f_MC[6] f_MC[7]
run {NMD}
"""


def run_np(nprocs: int):
    d = TMPROOT / f"8_10_np{nprocs}"
    shutil.rmtree(d, ignore_errors=True)
    log = run_lammps(hybrid_input(), d, nprocs=nprocs)
    acc = thermo_column(log, "f_MC[7]")[-1]
    natt = thermo_column(log, "f_MC[1]")[-1]
    conc = thermo_column(log, "f_MC[6]")[-1]
    return float(acc), float(natt), float(conc)


def test_highrank_consistency_and_acceptance():
    """np=16 (oversubscribed) vs np=1: every per-block consistency check must
    pass (the run errors out otherwise), and the acceptance ratios must agree
    within a loose stochastic band -- the viper 8B discrepancy was a factor
    ~80, far outside it."""
    if shutil.which(MPIRUN.split()[0]) is None:
        pytest.skip("mpirun not available")

    acc1, natt1, conc1 = run_np(1)
    acc16, natt16, conc16 = run_np(16)

    assert natt1 == NMD // NEVERY * NTRIALS
    assert natt16 == natt1

    # both runs completed => all `check yes` block verifications passed

    # loose band: same order of magnitude, absolute slack for small samples
    assert acc1 > 0.0 and acc16 > 0.0, (
        f"degenerate acceptance: np1 {acc1}, np16 {acc16} (mu {MU} off-band?)")
    lo, hi = 0.333, 3.0
    ratio = acc16 / acc1
    assert (lo < ratio < hi) or abs(acc16 - acc1) < 0.05, (
        f"acceptance rank-dependence: np1 {acc1:.4f} vs np16 {acc16:.4f} "
        f"(ratio {ratio:.2f}); viper-8B-class discrepancy would be ~80x")
