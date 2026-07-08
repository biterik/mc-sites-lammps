# Site-resolved Monte Carlo for LAMMPS — `compute sites/voronoi` + `fix mc/sites`

A LAMMPS feature for **occupancy Monte Carlo of an interstitial species on a dynamic,
geometry-derived site catalogue**, coupled to hybrid MD/MC. The candidate sites are the
vertices of the Voronoi tessellation of the atoms (the centers of the locally largest
empty spheres), so the catalogue adapts to the *current* structure — perfect crystals,
defects, voids, gas bubbles, surfaces — with no assumed sublattice. On that catalogue the
fix runs lattice-gas grand-canonical insertion/deletion (`gc`) or fixed-rate insertion
(`source`, irradiation-style).

Applications: hydrogen charging of metals (H in Ni is the reference case), He bubble
growth, and general lattice-gas sampling on sites that follow an evolving microstructure.
The feature is species- and potential-agnostic, with a fast path for the **GRACE**
machine-learning potentials.

> **This repository is a review package.** It is shared so colleagues can build the code,
> run the examples, and check the physics and usability **before** it is submitted as a
> pull request to LAMMPS. Feedback is welcome on everything — see
> [Open questions for reviewers](#open-questions-for-reviewers).

**Validation:** 43/43 local tests pass (geometry, Langmuir isotherm, round-trip energy
conservation, catalogue rebuild-invariance, a 2¹² brute-force enumeration cross-check,
hybrid MD/MC, source mode, fix stacking, and serial/MPI/BIGBIG consistency). Built and
tested against the GRACE fork `thermoatoms/lammps` @ `24da74cd` (LAMMPS **"11 Feb 2026"**),
Apple clang 21, Open MPI, serial and MPI, SMALLBIG and BIGBIG.

---

## Contents of this repository

| Path | What it is |
|---|---|
| `README.md` | this file — start here |
| `doc/` | the two LAMMPS manual pages (`fix mc/sites`, `compute sites/voronoi`) — full command reference and science |
| `examples/` | two small, fast runnable examples + reference logs + [`examples/README.md`](examples/README.md) walkthrough |
| `patches/` | the complete contribution as four `git am`-able patches (used by obtain-method B below) |
| `tests/` | the validation suite (LAMMPS inputs generated in-Python + numpy/pytest analysis) |
| `SPEC-MC-SITES.md` | the design specification (behavioral source of truth) |
| `LAMMPS-contributing-guide.md` | distilled LAMMPS contribution rules (for the eventual PR) |
| `ISSUE-DRAFT.md` | the upstream feature-proposal text |

The buildable LAMMPS source itself is **not** in this repository (it is a multi-GB tree);
you obtain it in Step 1 below.

---

## The science in brief

Grand-canonical MC of an interstitial species in a dense solid is hard for the standard
approach because inserting at a *random* position almost always lands in an atom's
repulsive core and is rejected. The physically relevant insertion points are the
**interstitial sites** — the roomy gaps — and in a real material those sites move and
change as the lattice relaxes and defects evolve.

`compute sites/voronoi` tessellates the atoms with Voro++ and takes the **Voronoi
vertices** as candidate sites; each vertex is the center of a locally largest empty sphere,
and its distance to the nearest atoms (its *clearance*) is the available insertion
clearance. Vertices are merged (`rmerge`) to remove lattice degeneracy and filtered by
clearance window, probe volume, region, coordination, or an occupancy veto.

`fix mc/sites` freezes a catalogue every `Nevery` steps (empty sites from the compute, plus
one occupied entry per current species atom) and runs symmetric lattice-gas trials with
acceptance `min(1, exp(-β(ΔU ∓ μ)))`, where ΔU is a full energy evaluation (required for
EAM/ACE/GRACE and any many-body potential).

### How it differs from the existing LAMMPS MC commands

| | insertion positions | move | reservoir / μ reference | best for |
|---|---|---|---|---|
| `fix gcmc` | **random** in a region | create/delete (+ optional MC moves) | ideal-gas: Λ, volume, 1/(N+1) | gases, dilute/porous systems |
| `fix atom/swap` | n/a (swaps identities) | type swap, **N conserved** | semi-grand Δμ | alloy composition |
| `fix widom` | random test insertions | **none** (measurement only) | — | measuring μ_excess |
| **`fix mc/sites`** | **only at Voronoi interstitial sites** | real create/delete | **lattice-gas** (no Λ/V/(N+1)) | interstitials in dense solids, moving microstructure |

Two points worth a reviewer's attention: (1) proposing only at geometrically favorable
sites raises acceptance dramatically in condensed phases — that is the point — but the
sampled ensemble is the lattice-gas on the discovered sites; (2) the fix's μ omits the
ideal-gas reservoir factors, so it differs from `fix gcmc`'s chemical potential by a
constant that must be **calibrated externally** if an absolute reference is needed (isotherm
shapes and phase plateaus do not depend on it). Full details are in
[`doc/fix_mc_sites.rst`](doc/fix_mc_sites.rst) and `SPEC-MC-SITES.md` §3.6.

---

## Getting started

Three steps: **obtain the source → compile → run**. Step 1 offers two equivalent ways to
obtain the code; pick one.

### Step 1 — Obtain the source code

#### Method A — clone the ready-made branch (quickest)

The contribution is already applied on a branch of a public GRACE fork. This gives you a
complete, buildable tree in one command:

```bash
git clone -b feature/mc-sites https://github.com/biterik/lammps.git lammps-mcsites
```

*(This fork must be public and pushed for the clone to work. If it is not yet available,
use Method B.)*

#### Method B — apply the patches onto the pinned fork commit (self-contained)

Uses only this repository plus the upstream GRACE fork; the exact base commit is pinned, so
the result is reproducible:

```bash
# 1. clone the GRACE fork and check out the exact base commit
git clone https://github.com/thermoatoms/lammps.git lammps-mcsites
cd lammps-mcsites
git checkout 24da74cd73323f5e7415fdd9a9670b88535464d3
git checkout -b feature/mc-sites

# 2. apply the four patches from THIS repository (adjust the path)
git am /path/to/this/repo/patches/00*.patch

# 3. confirm four commits were applied
git log --oneline -4
```

Both methods leave you in a LAMMPS tree on branch `feature/mc-sites` containing
`src/VORONOI/compute_sites_voronoi.{h,cpp}`, `src/MC/fix_mc_sites.{h,cpp}`, their doc pages,
and `examples/PACKAGES/mc_sites/`.

### Step 2 — Compile

From inside the LAMMPS tree (`lammps-mcsites`):

```bash
cmake -S cmake -B build \
  -D PKG_MC=on \
  -D PKG_VORONOI=on \
  -D DOWNLOAD_VORO=on \
  -D PKG_ML-PACE=on \
  -D NO_GRACE_TF=1 \
  -D BUILD_MPI=on \
  -D CMAKE_BUILD_TYPE=Release
cmake --build build -j
```

This produces `build/lmp`. For a serial build, drop `-D BUILD_MPI=on`. Voro++ is downloaded
automatically by `-D DOWNLOAD_VORO=on`.

> ⚠️ **Build gotcha (please read).** On this GRACE fork, plain `-D PKG_MC=on` **fails** with
> `pair_pace.h: file not found`, because the fork's `MC/fix_atom_swap.cpp` includes
> `pair_pace.h`. That is why `-D PKG_ML-PACE=on` is required above even though
> `fix mc/sites` itself does not use PACE, and `-D NO_GRACE_TF=1` lets ML-PACE build without
> TensorFlow (only needed to *run* GRACE, not to build or to run these examples/tests).
> **This coupling is an artifact of the fork, not of this contribution** — on upstream
> LAMMPS, `-D PKG_MC=on -D PKG_VORONOI=on` alone builds `fix mc/sites`.

Optional BIGBIG build (portability gate): add `-D LAMMPS_SIZES=bigbig` to a fresh build
directory.

### Step 3 — Run an example

```bash
cd examples/PACKAGES/mc_sites          # inside the LAMMPS tree
../../../build/lmp -in in.mc_sites.langmuir            # serial
# or:  mpirun -np 4 ../../../build/lmp -in in.mc_sites.langmuir
```

This is a non-interacting lattice gas (fcc host, `pair_style zero`, octahedral sites) at
μ = 0, where the site occupancy must converge to **θ = 0.5** — watch the `f_MC[6]` column.
Reference logs for serial and 4 ranks are committed next to the inputs.

A full walkthrough of both examples (including the output-vector meaning and the hybrid
MD/MC example) is in **[`examples/README.md`](examples/README.md)**. The same example files
are also in this repository under [`examples/`](examples/) for browsing without checking out
the LAMMPS tree.

---

## Running the validation suite

The tests live in this repository and drive your compiled `lmp` binary. Point them at it
with the `LMP_BIN` environment variable (and `MPIRUN` if your launcher differs):

```bash
python3 -m venv .venv && ./.venv/bin/pip install numpy pytest
LMP_BIN=/path/to/lammps-mcsites/build/lmp ./.venv/bin/python -m pytest tests/ -v
```

Roughly 43 checks, a few minutes. Every stochastic test records its RNG seed. Pinned test
dependencies are in `requirements-lock.txt`.

---

## Command reference

Full syntax, keywords, output, and restrictions are in the two manual pages:
[`doc/fix_mc_sites.rst`](doc/fix_mc_sites.rst) and
[`doc/compute_sites_voronoi.rst`](doc/compute_sites_voronoi.rst) (these are the same pages
that render in the LAMMPS HTML manual once the tree is built).

---

## GRACE and the energy-only fast path

`fix mc/sites` recomputes the total energy for every trial (`full_energy` style). For GRACE
this is the bottleneck, so the fix detects GRACE's energy-only switch via
`pair->extract("compute_energy_only")`, enables it for each MC block, and restores it
afterward — skipping the force computation in the TensorFlow graph. For every other pair
style it behaves exactly like `fix gcmc`. The handshake is written to compile on both the
fork (no upstream `ENERGY_ONLY` bit) and current upstream LAMMPS.

---

## Known limitations (v1)

Requires a 3d simulation, atom IDs, and per-type masses; reneighboring must be on. The MC
bookkeeping is serial (energy evaluation is parallel). Not supported in v1: Kokkos/GPU/INTEL
accelerated variants, ReaxFF and charge-equilibration potentials, core-shell models,
molecule insertion, and 2d. The static `file` site source assumes species atoms do not move
between blocks; for moving atoms use a dynamic compute catalogue with `occcut`.

---

## Open questions for reviewers

1. **The lattice-gas μ reference** (no ideal-gas factors) — is external calibration
   acceptable, or should the fix optionally emit an absolute μ?
2. **Ensemble interpretation** — does "propose only at Voronoi sites" match what you expect
   for your interstitial system?
3. **GRACE energy-only handshake** — is `extract("compute_energy_only")` the right interface
   to rely on long-term?
4. **Build experience** — did the ML-PACE coupling trip you up? Anything unclear here?
5. **Docs & examples** — is the physics clear from the manual pages and the example README
   alone?

Please send notes with your LAMMPS version and build flags.

---

## License

This contribution follows LAMMPS licensing: **GNU General Public License, version 2 only**
(GPL-2.0-only). Each source file carries the standard LAMMPS copyright/GPL header. See
`LICENSE`.
