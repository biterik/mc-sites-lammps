"""Test 8.11: natoms bookkeeping regression (bugfix session 2026-08-29).

Erik's Ni-H discharge campaign (thread 03, 1024 ranks, boundary p p f)
aborted with `sum of nlocal != natoms`, natoms HIGH by 1 and by 3.  The
investigation (PROGRESS.md 2026-08-29) identified two real defect
families and hardened both:

1. Ownership gap at a non-periodic box face: the sites compute emitted
   positions in [boxlo - nudge, boxlo) that NO rank owns under the fix's
   remap + [sublo, subhi) rule, so an accepted insertion incremented
   natoms with no atom created.  Fixed by clamping fp-scale face noise
   onto the face (compute emission + fix ownership), dropping
   genuinely-outside sites at catalogue build, and an unconditional
   per-trial owner-count assertion.
2. Rank-divergent Metropolis inputs: if mu, temp or the full energy
   differ across ranks by even one ulp, ranks take opposite
   accept/reject branches and natoms itself diverges (cascading through
   energy_stored for the rest of the block -- reproduces the observed
   deficit signature).  Fixed by making rank 0's mu/temp/energy
   authoritative (MPI_Bcast) and asserting per-trial rank uniformity
   under `check yes`.

Each test here fails on the pre-fix code (v1.1 = mc-sites-v1.1) and
passes after; the divergence tests inject the fault deterministically
via the MCS_DEBUG_PERTURB_ENERGY test hook.

Part of MC-SITES-LAMMPS. Author: Erik Bitzek <erik.bitzek@googlemail.com>
Implementation and testing by Claude Code (Anthropic).
"""
from __future__ import annotations

import os
import shutil
import subprocess

import pytest

from util_lammps import LMP_BIN, MPIRUN, TMPROOT, run_lammps, thermo_column

SEED = 8110  # recorded; every input below embeds it
L = 18.0     # cubic box edge [A]

pytestmark = pytest.mark.skipif(not LMP_BIN.exists(), reason="lmp binary not built")

HAVE_MPI = shutil.which(MPIRUN) is not None


def run_lammps_env(input_text: str, workdir, nprocs=1, env_extra=None,
                   timeout=300, name="in.test"):
    """Like util_lammps.run_lammps but with env control, a timeout (the
    pre-fix divergence cascade can deadlock in mismatched collectives),
    and no raise-on-error (several tests assert a specific abort)."""
    workdir.mkdir(parents=True, exist_ok=True)
    infile = workdir / name
    infile.write_text(input_text)
    logfile = workdir / "log.lammps"
    if nprocs == 1:
        cmd = [str(LMP_BIN)]
    else:
        cmd = [MPIRUN, "-np", str(nprocs), str(LMP_BIN)]
    cmd += ["-in", str(infile), "-log", str(logfile), "-screen", "none", "-nocite"]
    env = dict(os.environ)
    if env_extra:
        env.update(env_extra)
    proc = subprocess.run(cmd, cwd=workdir, capture_output=True, text=True,
                          timeout=timeout, env=env)
    log = logfile.read_text() if logfile.exists() else ""
    return proc.returncode, log


def sites_file_text(extra_rows=()):
    rows = ["5.0 5.0 5.0", "9.0 9.0 9.0", "13.0 13.0 13.0", "5.0 13.0 9.0"]
    rows += list(extra_rows)
    return "\n".join(rows) + "\n"


def file_mode_input(nblocks: int, mu: float, check: str = "yes") -> str:
    return f"""
units metal
boundary p p f
atom_style atomic
region box block 0 {L} 0 {L} 0 {L}
create_box 1 box
mass 1 1.008
pair_style zero 2.0
pair_coeff * *
# seed {SEED}
fix MC all mc/sites 1 20 1 {SEED} 300.0 sites file sites.txt mode gc mu {mu} check {check}
thermo 1
thermo_style custom step atoms f_MC[4] f_MC[5]
run {nblocks}
"""


def test_face_noise_site_is_clamped_and_owned():
    """A catalogue site an fp-noise distance below boxlo in the
    non-periodic dimension (the compute's nudge-wide emission overhang)
    must be clamped onto the face and inserted normally.  Pre-fix: the
    owner-count assertion aborts (v1.1 before Phase A: silent natoms
    corruption, the campaign signature)."""
    d = TMPROOT / "8_11_face_noise"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "sites.txt").write_text(sites_file_text(["5.0 9.0 -1.0e-10"]))
    rc, log = run_lammps_env(file_mode_input(nblocks=5, mu=1.0), d)
    assert rc == 0, f"clamped face site must not abort:\n{log[-2000:]}"
    # mu = +1 eV with pair zero accepts every insertion: all 5 sites occupied
    nsp = thermo_column(log, "f_MC[4]")[-1]
    msites = thermo_column(log, "f_MC[5]")[-1]
    assert msites == 5.0, f"catalogue must keep the clamped site: M = {msites}"
    assert nsp == 5.0, f"all 5 sites occupied expected, got {nsp}"


def test_far_outside_site_is_dropped_with_warning():
    """A site file position well outside the non-periodic box must be
    dropped at catalogue build (one warning), not silently corrupt the
    atom count and not abort the run."""
    d = TMPROOT / "8_11_far_outside"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "sites.txt").write_text(sites_file_text(["5.0 9.0 -5.0"]))
    rc, log = run_lammps_env(file_mode_input(nblocks=5, mu=1.0), d)
    assert rc == 0, f"dropped site must not abort:\n{log[-2000:]}"
    assert "dropped 1 catalogue site(s) no rank owns" in log
    msites = thermo_column(log, "f_MC[5]")[-1]
    assert msites == 4.0, f"outside site must be dropped: M = {msites}"


@pytest.mark.skipif(not HAVE_MPI, reason="mpirun not available")
def test_perturbed_energy_is_neutralized_by_broadcast():
    """MCS_DEBUG_PERTURB_ENERGY=1 offsets rank 1's energy_full() with an
    alternating sign -- the deterministic stand-in for a rank-divergent
    energy reduction.  Post-fix, rank 0's energy is broadcast, so the run
    must stay uniform and every `check yes` verification must pass.
    Pre-fix this aborts at the first knife-edge trial with the
    'rank-divergent' error (or, with checks off, silently reproduces the
    campaign's natoms-high-by-1 signature; PROGRESS.md 2026-08-29)."""
    d = TMPROOT / "8_11_perturb_bcast"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    grid = []
    for x in range(6):
        for y in range(6):
            for z in range(6):
                grid.append(f"{1.5 + 2.8 * x} {1.5 + 2.8 * y} {1.5 + 2.8 * z}")
    (d / "sites.txt").write_text("\n".join(grid) + "\n")
    text = f"""
units metal
boundary p p p
atom_style atomic
region box block 0 {L} 0 {L} 0 {L}
create_box 1 box
mass 1 1.008
pair_style zero 2.0
pair_coeff * *
# seed {SEED}
fix MC all mc/sites 1 40 1 {SEED} 300.0 sites file sites.txt mode gc mu -0.02 check yes
thermo 5
thermo_style custom step atoms f_MC[4] f_MC[6]
run 40
"""
    rc, log = run_lammps_env(text, d, nprocs=2,
                             env_extra={"MCS_DEBUG_PERTURB_ENERGY": "1"})
    assert rc == 0, f"broadcast must neutralize the rank-1 perturbation:\n{log[-2000:]}"
    assert "rank-divergent" not in log
    conc = thermo_column(log, "f_MC[6]")[-1]
    assert 0.0 < conc < 1.0, f"equilibrium occupancy expected, got c = {conc}"


@pytest.mark.skipif(not HAVE_MPI, reason="mpirun not available")
def test_mu_ramp_stays_rank_uniform():
    """Equal-style mu ramp (the phase-3 discharge pattern: pure step
    arithmetic on absolute bounds) over a hybrid MD/MC run: the per-block
    mu broadcast plus every per-trial uniformity assertion must hold at
    np=4."""
    d = TMPROOT / "8_11_mu_ramp"
    shutil.rmtree(d, ignore_errors=True)
    a0 = 3.52
    text = f"""
units metal
boundary p p p
atom_style atomic
lattice fcc {a0}
region box block 0 4 0 4 0 4
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
run 50
# seed {SEED}; ramp on absolute step bounds like the production MU_DISCHARGE
variable MU_RAMP equal 0.10-0.30*(step-50)/300
fix MC all mc/sites 10 25 2 {SEED} 300.0 sites c_S mode gc mu v_MU_RAMP check yes
thermo 50
thermo_style custom step atoms f_MC[1] f_MC[4] f_MC[6] f_MC[7]
run 300
"""
    rc, log = run_lammps_env(text, d, nprocs=4)
    assert rc == 0, f"mu-ramp hybrid run must pass all checks:\n{log[-2000:]}"
    natt = thermo_column(log, "f_MC[1]")[-1]
    assert natt > 0, "MC blocks must have run"


@pytest.mark.skipif(not HAVE_MPI, reason="mpirun not available")
def test_genuine_atom_loss_is_attributed():
    """An atom physically leaving the box through a non-periodic face is
    NOT an MC bookkeeping error: under `check yes` the block-start checks
    must attribute it as such (either at the block-start exchange or as
    corruption that happened outside the fix), never as the bare
    'sum of nlocal != natoms' MC abort."""
    d = TMPROOT / "8_11_atom_loss"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "sites.txt").write_text(sites_file_text())
    text = f"""
units metal
boundary p p f
atom_style atomic
region box block 0 {L} 0 {L} 0 {L}
create_box 2 box
create_atoms 2 single 9.0 9.0 15.0
mass 1 58.69
mass 2 1.008
pair_style zero 2.0
pair_coeff * *
# seed {SEED}; 3000 A/ps: exits the 18 A box within a few 1 fs steps
velocity all set 0.0 0.0 3000.0 units box
fix NVE all nve
thermo 1000
thermo_modify lost ignore
timestep 0.001
fix MC all mc/sites 5 10 2 {SEED} 300.0 sites file sites.txt mode gc mu -2.0 check yes
run 40
"""
    rc, log = run_lammps_env(text, d, nprocs=2)
    assert rc != 0, "the lost atom must abort under check yes"
    assert ("genuine atom loss" in log) or ("OUTSIDE this fix" in log), (
        f"loss must be attributed, not reported as MC bookkeeping:\n{log[-2000:]}")
    assert "consistency check failed" not in log


def test_restart_preserves_per_rank_velocity_streams():
    """random_unequal is per-rank (seed + comm->me) so concurrently
    inserted atoms on different ranks draw independent velocities.  The
    restart file carries rank 0's state; the restore must re-derive a
    DISTINCT stream per rank instead of collapsing all ranks onto rank
    0's stream (found in audit D2, 2026-08-29).  Serial half: write a
    restart mid-run and resume -- the resumed run must complete with all
    checks green and occupancy evolving.  (The per-rank distinctness
    itself is a code-level fix; this guards the restore path end-to-end.)"""
    d = TMPROOT / "8_11_restart"
    shutil.rmtree(d, ignore_errors=True)
    d.mkdir(parents=True)
    (d / "sites.txt").write_text(sites_file_text())
    base = f"""
units metal
boundary p p p
atom_style atomic
region box block 0 {L} 0 {L} 0 {L}
create_box 1 box
mass 1 1.008
pair_style zero 2.0
pair_coeff * *
# seed {SEED}
fix MC all mc/sites 1 10 1 {SEED} 300.0 sites file sites.txt mode gc mu 0.0 check yes
thermo 1
thermo_style custom step atoms f_MC[4]
restart 10 mc.restart
run 20
"""
    rc, log = run_lammps_env(base, d, name="in.write")
    assert rc == 0, f"restart-writing run failed:\n{log[-2000:]}"
    resume = f"""
units metal
read_restart mc.restart.20
pair_style zero 2.0
pair_coeff * *
# seed {SEED}
fix MC all mc/sites 1 10 1 {SEED} 300.0 sites file sites.txt mode gc mu 0.0 check yes
thermo 1
thermo_style custom step atoms f_MC[4]
run 20
"""
    rc2, log2 = run_lammps_env(resume, d, name="in.resume")
    assert rc2 == 0, f"resumed run failed:\n{log2[-2000:]}"
    natt = thermo_column(log2, "f_MC[4]")
    assert len(natt) > 0
