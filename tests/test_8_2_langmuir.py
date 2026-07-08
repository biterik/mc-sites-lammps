"""SPEC test 8.2: Langmuir isotherm (fix mc/sites, static file sites, pair zero).

With no interactions (pair_style zero) the lattice-gas GC occupancy must
follow theta(mu,T) = 1/(1+exp(-mu/kBT)). Nine (mu,T) points at 300 K and
600 K on a 216-site list; tolerance |theta_MC - theta| < 0.02 per SPEC.
Extreme-mu guards: mu -> +-1 eV forces theta -> 1/0.

All runs record their RNG seed in the input file.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from util_lammps import LMP_BIN, TMPROOT, run_lammps, thermo_column

KB = 8.617333262e-5    # eV/K (LAMMPS metal units value)
M_SITES = 216

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def write_sites_file(path, n_side=6, spacing=3.0, offset=1.5):
    with open(path, "w") as f:
        f.write("# sc site grid for Langmuir test\n")
        for i in range(n_side):
            for j in range(n_side):
                for k in range(n_side):
                    f.write(f"{offset + i * spacing:.6f} {offset + j * spacing:.6f} "
                            f"{offset + k * spacing:.6f}\n")


def langmuir_input(mu: float, T: float, seed: int, n_equil: int, n_sample: int) -> str:
    L = 18.0
    return f"""
units metal
boundary p p p
atom_style atomic
region box block 0 {L} 0 {L} 0 {L}
create_box 1 box
mass 1 1.008
pair_style zero 2.0
pair_coeff * *
# seed {seed}
fix MC all mc/sites 1 50 1 {seed} {T} sites file sites.txt mode gc mu {mu}
thermo 1
thermo_style custom step atoms f_MC[4] f_MC[5] f_MC[7]
run {n_equil}
run {n_sample}
"""


def run_point(name: str, mu: float, T: float, seed: int,
              n_equil: int = 400, n_sample: int = 1600) -> float:
    d = TMPROOT / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    write_sites_file(d / "sites.txt")
    log = run_lammps(langmuir_input(mu, T, seed, n_equil, n_sample), d)
    n = thermo_column(log, "Atoms")
    # last run section = sampling: take the final n_sample entries
    samples = np.array(n[-(n_sample):], dtype=float)
    return float(samples.mean()) / M_SITES


def theta_exact(mu: float, T: float) -> float:
    return 1.0 / (1.0 + math.exp(-mu / (KB * T)))


@pytest.mark.parametrize(
    "mu,T,seed",
    [
        (-0.10, 300.0, 1001),
        (-0.04, 300.0, 1002),
        (-0.02, 300.0, 1003),
        (0.00, 300.0, 1004),
        (+0.02, 300.0, 1005),
        (+0.10, 300.0, 1006),
        (-0.05, 600.0, 1007),
        (+0.05, 600.0, 1008),
        (+0.15, 600.0, 1009),
    ],
)
def test_langmuir_point(mu, T, seed):
    th_mc = run_point(f"8_2_mu{mu:+.2f}_T{int(T)}", mu, T, seed)
    th = theta_exact(mu, T)
    assert abs(th_mc - th) < 0.02, (
        f"mu={mu} T={T}: theta_MC={th_mc:.4f} vs exact {th:.4f}")
    print(f"mu={mu:+.2f} T={T:.0f}: theta_MC={th_mc:.4f} exact={th:.4f} "
          f"|d|={abs(th_mc - th):.4f}")


def test_extreme_mu_guards():
    th_lo = run_point("8_2_extreme_lo", -1.0, 300.0, 2001, n_equil=100, n_sample=200)
    assert th_lo < 0.01, f"mu=-1: theta={th_lo}"
    th_hi = run_point("8_2_extreme_hi", +1.0, 300.0, 2002, n_equil=100, n_sample=200)
    assert th_hi > 0.99, f"mu=+1: theta={th_hi}"
