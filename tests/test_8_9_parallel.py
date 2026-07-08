"""SPEC test 8.9: parallel / portability gates.

Serial vs mpirun -np {2,4}: statistically consistent isotherm points
(file and dynamic catalogues) and deterministic source-mode placement
with a fixed seed. The BIGBIG binary (lammps/build-bigbig/lmp), when
present, must pass a Langmuir point too; SMALLBIG is the default build
exercised by the whole suite.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import math
import shutil

import numpy as np
import pytest

from util_lammps import LMP_BIN, REPO, TMPROOT, run_lammps, thermo_column

KB = 8.617333262e-5
A0 = 3.52
NCELL = 3
NHOST = 4 * NCELL**3
N_OCT = 4 * NCELL**3
BIGBIG_BIN = REPO / "lammps" / "build-bigbig" / "lmp"

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


def gc_input(mu, T, seed, sites_arg, n_equil, n_sample):
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
fix MC all mc/sites 1 40 2 {seed} {T} sites {sites_arg} mode gc mu {mu}
thermo 1
thermo_style custom step atoms
run {n_equil}
run {n_sample}
"""


def theta_of_run(d, log, n_sample):
    n = np.array(thermo_column(log, "Atoms")[-n_sample:], dtype=float)
    return float((n - NHOST).mean()) / N_OCT


@pytest.mark.parametrize("sites_arg", ["file sites.txt", "c_S"])
@pytest.mark.parametrize("nprocs", [2, 4])
def test_isotherm_consistent_across_ranks(sites_arg, nprocs):
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    T, mu = 300.0, 0.0
    n_equil, n_sample = 300, 1200
    tag = "file" if sites_arg.startswith("file") else "dyn"
    results = {}
    for np_i in (1, nprocs):
        d = TMPROOT / f"8_9_{tag}_np{np_i}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        if tag == "file":
            write_oct_sites(d / "sites.txt")
        log = run_lammps(gc_input(mu, T, 9000 + np_i, sites_arg, n_equil, n_sample),
                         d, nprocs=np_i)
        results[np_i] = theta_of_run(d, log, n_sample)
    th = 1.0 / (1.0 + math.exp(-mu / (KB * T)))
    for np_i, val in results.items():
        assert abs(val - th) < 0.02, f"np={np_i} ({tag}): theta {val:.4f} vs {th:.4f}"
    assert abs(results[1] - results[nprocs]) < 0.03, (
        f"{tag}: serial {results[1]:.4f} vs np{nprocs} {results[nprocs]:.4f}")
    print(f"{tag} np{nprocs}: serial {results[1]:.4f}, parallel {results[nprocs]:.4f}")


def source_input(seed, nsteps):
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
# seed {seed}
fix MC all mc/sites 1 0 2 {seed} 300.0 sites file sites.txt mode source rate 4
dump D all custom 1 dump.atoms id type x y z
dump_modify D format float %.12g sort id
thermo 1
thermo_style custom step atoms
run {nsteps}
"""


def last_species_positions(d, natoms_final):
    text = (d / "dump.atoms").read_text().splitlines()
    idx = len(text) - 1 - text[::-1].index("ITEM: ATOMS id type x y z")
    rows = np.array([list(map(float, line.split()))
                     for line in text[idx + 1: idx + 1 + natoms_final]])
    species = rows[rows[:, 1] == 2][:, 2:5]
    return species[np.lexsort(np.round(species, 6).T)]


@pytest.mark.parametrize("nprocs", [2, 4])
def test_source_mode_deterministic_across_ranks(nprocs):
    """Fixed seed: source-mode placement must be identical serial vs MPI."""
    if shutil.which("mpirun") is None:
        pytest.skip("no mpirun")
    nsteps, seed = 6, 9100
    pos = {}
    for np_i in (1, nprocs):
        d = TMPROOT / f"8_9_source_np{np_i}"
        shutil.rmtree(d, ignore_errors=True)
        d.mkdir(parents=True)
        write_oct_sites(d / "sites.txt")
        log = run_lammps(source_input(seed, nsteps), d, nprocs=np_i)
        atoms = np.array(thermo_column(log, "Atoms"), dtype=int)
        assert atoms[-1] == NHOST + 4 * nsteps
        pos[np_i] = last_species_positions(d, atoms[-1])
    assert np.allclose(pos[1], pos[nprocs], atol=1e-9), (
        f"source placement differs between serial and np{nprocs}")


def test_bigbig_langmuir_point():
    """The -DLAMMPS_SIZES=bigbig build must pass a Langmuir point (test 8.2)."""
    if not BIGBIG_BIN.exists():
        pytest.skip("bigbig binary not built")
    import os
    import subprocess
    T, mu, seed = 300.0, 0.0, 9200
    n_equil, n_sample = 300, 1200
    d = TMPROOT / "8_9_bigbig"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    write_oct_sites(d / "sites.txt")
    inp = d / "in.test"
    inp.write_text(gc_input(mu, T, seed, "file sites.txt", n_equil, n_sample))
    logf = d / "log.lammps"
    proc = subprocess.run([str(BIGBIG_BIN), "-in", str(inp), "-log", str(logf),
                           "-screen", "none", "-nocite"], cwd=d, capture_output=True,
                          text=True)
    assert proc.returncode == 0, f"bigbig run failed: {proc.stderr[-1500:]}"
    th = theta_of_run(d, logf.read_text(), n_sample)
    assert abs(th - 0.5) < 0.02, f"bigbig theta {th:.4f} vs 0.5"
    print(f"bigbig theta = {th:.4f}")
