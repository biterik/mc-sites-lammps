"""SPEC test 8.4: rebuild-invariance (detailed-balance quantification).

Static frozen host (no integration fix) + dynamic `sites c_ID` catalogue:
rebuilding the catalogue every MC block must reproduce the static
file-list isotherm within MC error, and (pair zero) both must match the
analytic Langmuir curve. rmin 1.6 selects the 108 octahedral sites of
the 3x3x3 fcc box; occupied sites vanish from the tessellation
geometrically (the species atom's own small vertices fall below rmin).

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
NSITES = 4 * NCELL**3    # 108 octahedral

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def write_oct_sites(path):
    basis = [(0.5, 0.0, 0.0), (0.0, 0.5, 0.0), (0.0, 0.0, 0.5), (0.5, 0.5, 0.5)]
    with open(path, "w") as f:
        f.write("# fcc octahedral sites 3x3x3\n")
        for i in range(NCELL):
            for j in range(NCELL):
                for k in range(NCELL):
                    for bx, by, bz in basis:
                        f.write(f"{(i + bx) * A0:.6f} {(j + by) * A0:.6f} "
                                f"{(k + bz) * A0:.6f}\n")


def isotherm_input(mu: float, T: float, seed: int, sites_arg: str,
                   n_equil: int, n_sample: int) -> str:
    compute_line = ""
    if sites_arg.startswith("c_"):
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
fix MC all mc/sites 1 40 2 {seed} {T} sites {sites_arg} mode gc mu {mu}
thermo 1
thermo_style custom step atoms f_MC[4] f_MC[5]
run {n_equil}
run {n_sample}
"""


def run_isotherm_point(name: str, mu: float, T: float, seed: int, sites_arg: str,
                       n_equil: int = 300, n_sample: int = 1200) -> float:
    d = TMPROOT / name
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    if sites_arg.startswith("file"):
        write_oct_sites(d / "sites.txt")
    log = run_lammps(isotherm_input(mu, T, seed, sites_arg, n_equil, n_sample), d)
    n = np.array(thermo_column(log, "Atoms")[-n_sample:], dtype=float)
    return float((n - NHOST).mean()) / NSITES


def theta_exact(mu: float, T: float) -> float:
    return 1.0 / (1.0 + math.exp(-mu / (KB * T)))


@pytest.mark.parametrize("mu,seed", [(-0.02, 3101), (0.00, 3102), (+0.02, 3103)])
def test_dynamic_matches_static_and_analytic(mu, seed):
    T = 300.0
    th_file = run_isotherm_point(f"8_4_file_mu{mu:+.2f}", mu, T, seed, "file sites.txt")
    th_dyn = run_isotherm_point(f"8_4_dyn_mu{mu:+.2f}", mu, T, seed + 50, "c_S")
    th = theta_exact(mu, T)
    assert abs(th_file - th) < 0.02, f"file: {th_file:.4f} vs analytic {th:.4f}"
    assert abs(th_dyn - th) < 0.02, f"dynamic: {th_dyn:.4f} vs analytic {th:.4f}"
    assert abs(th_dyn - th_file) < 0.02, (
        f"rebuild changed the isotherm: dynamic {th_dyn:.4f} vs file {th_file:.4f}")
    print(f"mu={mu:+.2f}: file={th_file:.4f} dynamic={th_dyn:.4f} analytic={th:.4f}")


def test_dynamic_catalogue_size_is_conserved():
    """On the static lattice the catalogue must always hold all 108 sites
    (empty + occupied) regardless of occupancy."""
    d = TMPROOT / "8_4_M_conserved"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    log = run_lammps(isotherm_input(0.0, 300.0, 3200, "c_S", 100, 300), d)
    m = np.array(thermo_column(log, "f_MC[5]")[5:], dtype=float)
    assert np.all(m == NSITES), (
        f"catalogue size varied: min {m.min()}, max {m.max()}, expected {NSITES}")
