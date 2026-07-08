"""SPEC test 8.5: exact-enumeration cross-check with interactions.

12 sites on a ring (NN chord 3.0 A, 2nd-neighbor 5.80 A > cutoff 4.0 A)
in a host-free box; inserted atoms interact via lj/cut, so the system is
exactly the MC-DRIVER NN lattice gas with J = U_lj(3.0). Exact <c> and
P(N) from direct 2^12 enumeration with the same energies LAMMPS computes
(the lj/cut formula is verified against a LAMMPS run at full occupancy);
the MC sampler must reproduce P(N) within total-variation 0.05 and <c>
within 0.015 (methodology of MC-DRIVER tests/test_enumeration.py).

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
M = 12
EPS_LJ = 0.05
SIGMA = 2.67
CUTOFF = 4.0
NN_DIST = 3.0
T_K = 300.0
MU = -0.04
SEED = 8501
BOXL = 30.0

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def ring_coords() -> np.ndarray:
    R = NN_DIST / (2.0 * math.sin(math.pi / M))
    c = BOXL / 2.0
    pts = [(c + R * math.cos(2 * math.pi * i / M),
            c + R * math.sin(2 * math.pi * i / M), c) for i in range(M)]
    return np.array(pts)


def lj(r: float) -> float:
    if r >= CUTOFF:
        return 0.0
    sr6 = (SIGMA / r) ** 6
    return 4.0 * EPS_LJ * (sr6 * sr6 - sr6)


def pair_energy_matrix(coords: np.ndarray) -> np.ndarray:
    J = np.zeros((M, M))
    for i in range(M):
        for j in range(i + 1, M):
            r = float(np.linalg.norm(coords[i] - coords[j]))
            J[i, j] = J[j, i] = lj(r)
    return J


def enumerate_exact(J: np.ndarray, mu: float, T: float):
    beta = 1.0 / (KB * T)
    args = np.empty(1 << M)
    ns = np.empty(1 << M, dtype=int)
    for s in range(1 << M):
        bits = [(s >> i) & 1 for i in range(M)]
        n = sum(bits)
        u = 0.0
        for i in range(M):
            if bits[i]:
                for j in range(i + 1, M):
                    if bits[j]:
                        u += J[i, j]
        args[s] = -beta * (u - mu * n)
        ns[s] = n
    w = np.exp(args - args.max())
    w /= w.sum()
    P = np.zeros(M + 1)
    for s in range(1 << M):
        P[ns[s]] += w[s]
    mean_c = float(sum(n * P[n] for n in range(M + 1))) / M
    return mean_c, P


def write_sites(path, coords):
    with open(path, "w") as f:
        f.write("# 12-site ring\n")
        for x, y, z in coords:
            f.write(f"{x:.10f} {y:.10f} {z:.10f}\n")


BASE = f"""
units metal
boundary p p p
atom_style atomic
region box block 0 {BOXL} 0 {BOXL} 0 {BOXL}
create_box 1 box
mass 1 1.008
pair_style lj/cut {CUTOFF}
pair_coeff 1 1 {EPS_LJ} {SIGMA}
"""


def test_lammps_energy_matches_analytic():
    """Full ring occupancy: LAMMPS pe must equal the analytic sum."""
    coords = ring_coords()
    J = pair_energy_matrix(coords)
    e_full_analytic = float(np.triu(J, 1).sum())
    d = TMPROOT / "8_5_efull"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    create = "\n".join(
        f"create_atoms 1 single {x:.10f} {y:.10f} {z:.10f} units box" for x, y, z in coords)
    log = run_lammps(BASE + create + """
thermo_style custom step atoms pe
thermo_modify format float %.15g
run 0
""", d)
    pe = thermo_column(log, "PotEng")[0]
    assert abs(pe - e_full_analytic) < 1e-9, (
        f"LAMMPS pe {pe:.12g} vs analytic {e_full_analytic:.12g}")
    print(f"E_full: LAMMPS {pe:.12g} == analytic {e_full_analytic:.12g}")


def test_mc_matches_enumeration():
    coords = ring_coords()
    J = pair_energy_matrix(coords)
    exact_c, exact_P = enumerate_exact(J, MU, T_K)

    d = TMPROOT / "8_5_mc"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    write_sites(d / "sites.txt", coords)
    n_equil, n_sample = 3000, 15000
    log = run_lammps(BASE + f"""
# seed {SEED}
fix MC all mc/sites 1 20 1 {SEED} {T_K} sites file sites.txt mode gc mu {MU}
thermo 1
thermo_style custom step atoms
run {n_equil}
run {n_sample}
""", d)
    n = np.array(thermo_column(log, "Atoms")[-n_sample:], dtype=int)
    mc_c = float(n.mean()) / M
    mc_P = np.bincount(n, minlength=M + 1)[: M + 1] / len(n)

    tv = 0.5 * float(np.abs(mc_P - exact_P).sum())
    print(f"<c>: MC {mc_c:.4f} exact {exact_c:.4f}; P(N) TV distance {tv:.4f}")
    assert abs(mc_c - exact_c) < 0.015, f"<c> MC {mc_c:.4f} vs exact {exact_c:.4f}"
    assert tv < 0.05, f"P(N) total-variation distance {tv:.4f} >= 0.05"
