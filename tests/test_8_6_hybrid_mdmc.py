"""SPEC test 8.6: hybrid MD/MC smoke test.

lj/cut fcc host at low T + NVT integration between MC blocks + dynamic
sites/voronoi catalogue. Requirements: runs stably (no lost atoms, bounded
temperature and total energy), and c(mu) is monotone over 3 mu points.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import shutil

import numpy as np
import pytest

from util_lammps import LMP_BIN, TMPROOT, run_lammps, thermo_column

A0 = 3.52
NCELL = 3
NHOST = 4 * NCELL**3
T_MD = 30.0

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def hybrid_input(mu: float, seed: int, nsteps: int) -> str:
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
pair_coeff 1 1 0.20 2.30
pair_coeff 1 2 0.02 1.50
pair_coeff 2 2 0.02 1.00
timestep 0.0005
# seed {seed}
velocity all create {T_MD} {seed} mom yes rot yes
compute S all sites/voronoi rmerge 0.3 rmin 1.55 rmax 2.1
fix INT all nvt temp {T_MD} {T_MD} 0.05
fix MC all mc/sites 10 20 2 {seed + 1} 300.0 sites c_S mode gc mu {mu}
thermo 10
thermo_style custom step atoms temp etotal f_MC[4] f_MC[5] f_MC[7]
run {nsteps}
"""


def run_hybrid(mu: float, seed: int, nsteps: int = 2000):
    d = TMPROOT / f"8_6_mu{mu:+.2f}"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    log = run_lammps(hybrid_input(mu, seed, nsteps), d)
    atoms = np.array(thermo_column(log, "Atoms"), dtype=float)
    temp = np.array(thermo_column(log, "Temp"), dtype=float)
    etot = np.array(thermo_column(log, "TotEng"), dtype=float)
    m = np.array(thermo_column(log, "f_MC[5]"), dtype=float)
    return atoms, temp, etot, m


@pytest.mark.parametrize("mu,seed", [(-0.10, 8601), (-0.02, 8602), (+0.06, 8603)])
def test_hybrid_runs_stably(mu, seed):
    atoms, temp, etot, m = run_hybrid(mu, seed)
    # stability: bounded T, no NaN/blow-up in total energy, hosts preserved
    assert np.all(np.isfinite(etot)), "total energy diverged"
    assert np.all(atoms >= NHOST), "host atoms lost"
    half = len(temp) // 2
    assert temp[half:].mean() < 5 * T_MD, (
        f"temperature ran away: <T>={temp[half:].mean():.1f} K")
    # catalogue stays populated on the (thermally fluctuating) lattice
    assert np.all(m[2:] > 0), "dynamic catalogue collapsed to zero sites"


def test_c_monotone_in_mu():
    cs = {}
    for mu, seed in [(-0.10, 8601), (-0.02, 8602), (+0.06, 8603)]:
        atoms, _, _, m = run_hybrid(mu, seed)
        half = len(atoms) // 2
        n_species = atoms[half:] - NHOST
        m_mean = m[half:].mean()
        cs[mu] = float(n_species.mean()) / m_mean
    print(f"c(mu): {cs}")
    assert cs[-0.10] < cs[-0.02] < cs[+0.06], f"c(mu) not monotone: {cs}"
