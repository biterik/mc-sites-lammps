"""SPEC test 8.1: compute sites/voronoi geometry unit test.

Ideal fcc, a = 3.52 A, pair_style zero, no MD. Analytic targets:
octahedral sites (4/cell) at clearance a/2 = 1.760 A; tetrahedral
sites (8/cell) at clearance sqrt(3)*a/4 = 1.524197 A. With rmerge 0.3
the compute must find exactly 4+8 merged sites/cell; window rmin 1.6
selects exactly the 4 octahedral. Perturbation (0.05 A, fixed seed
12345), triclinic variant, surface slab + coord criterion, and a void
variant are covered per SPEC 8.1. MPI consistency (2 and 4 ranks) is
checked on the base case.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from util_lammps import LMP_BIN, TMPROOT, parse_dump_local, run_lammps

A0 = 3.52
R_OCT = A0 / 2.0                      # 1.760
R_TET = math.sqrt(3.0) * A0 / 4.0     # 1.524197

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")


def fcc_input(ncell: int = 3, extra_compute: str = "", perturb: str = "",
              triclinic: bool = False, region_and_atoms: str = "") -> str:
    if not region_and_atoms:
        if triclinic:
            region_and_atoms = f"""
region box prism 0 {ncell} 0 {ncell} 0 {ncell} 1 0 0
create_box 1 box
create_atoms 1 box
"""
        else:
            region_and_atoms = f"""
region box block 0 {ncell} 0 {ncell} 0 {ncell}
create_box 1 box
create_atoms 1 box
"""
    return f"""
units metal
boundary p p p
atom_style atomic
lattice fcc {A0}
{region_and_atoms}
mass 1 58.69
pair_style zero 4.0
pair_coeff * *
{perturb}
compute S all sites/voronoi rmerge 0.3 {extra_compute}
dump D all local 1 dump.sites c_S[1] c_S[2] c_S[3] c_S[4] c_S[5]
dump_modify D format float %.12g
thermo_style custom step atoms c_S
run 0
"""


def site_rows(workdir) -> np.ndarray:
    frames = parse_dump_local(workdir / "dump.sites")
    return frames[0]


def classify(rows: np.ndarray) -> tuple[int, int, int]:
    """Return (n_oct, n_tet, n_other) by clearance windows."""
    r = rows[:, 3]
    n_oct = int(np.sum(np.abs(r - R_OCT) < 0.12))
    n_tet = int(np.sum(np.abs(r - R_TET) < 0.12))
    return n_oct, n_tet, len(r) - n_oct - n_tet


def test_ideal_fcc_counts_and_clearances():
    d = TMPROOT / "8_1_ideal"
    shutil.rmtree(d, ignore_errors=True)
    run_lammps(fcc_input(ncell=3), d)
    rows = site_rows(d)
    ncell3 = 27
    assert rows.shape[0] == 12 * ncell3, f"expected 324 sites, got {rows.shape[0]}"
    n_oct, n_tet, n_other = classify(rows)
    assert n_oct == 4 * ncell3, f"oct: {n_oct} != 108"
    assert n_tet == 8 * ncell3, f"tet: {n_tet} != 216"
    assert n_other == 0
    # clearances exact on the ideal lattice
    r = rows[:, 3]
    oct_r = r[np.abs(r - R_OCT) < 0.12]
    tet_r = r[np.abs(r - R_TET) < 0.12]
    assert np.max(np.abs(oct_r - R_OCT)) < 1e-6
    assert np.max(np.abs(tet_r - R_TET)) < 1e-6


def test_window_selects_octahedral_only():
    d = TMPROOT / "8_1_window"
    shutil.rmtree(d, ignore_errors=True)
    run_lammps(fcc_input(ncell=3, extra_compute="rmin 1.6"), d)
    rows = site_rows(d)
    assert rows.shape[0] == 108, f"rmin 1.6: expected 108 oct sites, got {rows.shape[0]}"
    assert np.max(np.abs(rows[:, 3] - R_OCT)) < 1e-6


def test_perturbed_lattice_same_counts():
    """0.05 A random displacement (seed 12345): merging must keep 12/cell."""
    d = TMPROOT / "8_1_perturbed"
    shutil.rmtree(d, ignore_errors=True)
    run_lammps(fcc_input(ncell=3,
                         perturb="displace_atoms all random 0.05 0.05 0.05 12345 units box"), d)
    rows = site_rows(d)
    assert rows.shape[0] == 324, f"perturbed: expected 324 sites, got {rows.shape[0]}"
    n_oct, n_tet, n_other = classify(rows)
    assert n_oct == 108 and n_tet == 216 and n_other == 0, (n_oct, n_tet, n_other)


def test_triclinic_same_counts():
    """Same fcc crystal in a prism box (xy tilt = one lattice vector)."""
    d = TMPROOT / "8_1_triclinic"
    shutil.rmtree(d, ignore_errors=True)
    run_lammps(fcc_input(ncell=3, triclinic=True), d)
    rows = site_rows(d)
    assert rows.shape[0] == 324, f"triclinic: expected 324 sites, got {rows.shape[0]}"
    n_oct, n_tet, n_other = classify(rows)
    assert n_oct == 108 and n_tet == 216 and n_other == 0, (n_oct, n_tet, n_other)


def test_surface_slab_coord_criterion():
    """Slab with vacuum in z: coord 4 2.0 must eliminate vacuum sites."""
    d = TMPROOT / "8_1_slab"
    shutil.rmtree(d, ignore_errors=True)
    region_and_atoms = """
region box block 0 3 0 3 -3 6
create_box 1 box
region slab block INF INF INF INF 0 2.9
create_atoms 1 region slab
"""
    inp = fcc_input(extra_compute="coord 4 2.0",
                    region_and_atoms=region_and_atoms).replace(
        "boundary p p p", "boundary p p f")
    run_lammps(inp, d)
    rows = site_rows(d)
    assert rows.shape[0] > 0, "slab: no sites found at all"
    z = rows[:, 2]
    zmax_atoms = 2.75 * A0    # last atom layer of the slab
    assert np.max(z) < zmax_atoms + 2.0, (
        f"vacuum site at z={np.max(z):.3f} (atoms end at {zmax_atoms:.3f})")
    assert np.min(z) > -2.0, f"vacuum site below slab at z={np.min(z):.3f}"
    # every accepted site indeed reports >= 4 host atoms (coord column)
    assert np.min(rows[:, 4]) >= 4


def test_void_interior_sites():
    """Removing a cluster of atoms creates interior sites with large clearance."""
    d = TMPROOT / "8_1_void"
    shutil.rmtree(d, ignore_errors=True)
    perturb = """
region hole sphere 2 2 2 1.1
delete_atoms region hole compress yes
"""
    run_lammps(fcc_input(ncell=4, perturb=perturb), d)
    rows = site_rows(d)
    center = 2 * A0
    d2 = np.sum((rows[:, :3] - center) ** 2, axis=1)
    near = rows[d2 < (1.1 * A0) ** 2]
    assert near.shape[0] > 0, "no sites inside the void"
    assert np.max(near[:, 3]) > 2.0, (
        f"void interior clearance max {np.max(near[:, 3]):.3f} <= 2.0")
    print(f"void variant: {rows.shape[0]} total sites, "
          f"{near.shape[0]} in void, max clearance {np.max(near[:, 3]):.4f}")


@pytest.mark.parametrize("nprocs", [2, 4])
def test_mpi_consistency(nprocs):
    """Serial and MPI runs must find the identical site catalogue."""
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    d1 = TMPROOT / "8_1_serial_ref"
    shutil.rmtree(d1, ignore_errors=True)
    run_lammps(fcc_input(ncell=3), d1)
    ref = site_rows(d1)
    dn = TMPROOT / f"8_1_mpi{nprocs}"
    shutil.rmtree(dn, ignore_errors=True)
    run_lammps(fcc_input(ncell=3), dn, nprocs=nprocs)
    par = site_rows(dn)
    assert par.shape[0] == ref.shape[0], (
        f"np={nprocs}: {par.shape[0]} sites vs serial {ref.shape[0]}")
    # same positions as sets (order differs across ranks; sort on rounded
    # coordinates so fp noise ~1e-16 cannot flip tie-breaking)
    ref_sorted = ref[np.lexsort(np.round(ref[:, :3], 6).T)]
    par_sorted = par[np.lexsort(np.round(par[:, :3], 6).T)]
    assert np.allclose(ref_sorted[:, :4], par_sorted[:, :4], atol=1e-6), (
        "site catalogues differ between serial and MPI")
