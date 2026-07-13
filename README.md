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

## Usage — controlling *where* atoms are inserted

The feature is two cooperating commands. **`compute sites/voronoi` decides *where* the
candidate sites are** — every geometric possibility (size window, region, coordination,
occupancy) is a filter keyword on the compute. **`fix mc/sites` decides *whether and how*
atoms are inserted or deleted** on those sites (chemical potential, rate, region, charge,
caps). The minimal pattern is:

```lammps
compute SITES all sites/voronoi rmerge 0.3                                # find + filter candidate sites
fix     MC    all mc/sites 100 200 2 12345 500.0 sites c_SITES mode gc mu -2.40   # run GCMC on them
```

`rmerge` is the only required argument: Voronoi vertices of a high-symmetry lattice split
into near-coincident copies under thermal noise, so vertices closer than `rmerge` are
merged to one site (its clearance = the minimum over the cluster). Use `rmerge` well below
the spacing between distinct sites — **0.3 Å works for metals**. Everything else below is
optional; **a site is kept only if it passes _all_ active criteria**, so you combine them
freely.

### 1. Select sites by size (two independent geometric classifiers)

Each Voronoi vertex is the center of a locally largest empty sphere. You can bound sites by
**either** the sphere radius **or** a probe volume — these are different measures of "how
much room is here," and you may use one or both.

| What you bound | Keywords | Units | Notes |
|---|---|---|---|
| **Clearance** = empty-sphere radius (distance to nearest atom) | `rmin <d>` / `rmax <d>` | distance | always computed; the cheap, default classifier |
| **Probe Voronoi volume** (volume of the Voronoi cell a test point at the site would have among the atoms) | `vmin <v>` / `vmax <v>` | volume | matches the von Pezold *et al.* volume classification (Acta Mater. **59**, 2969, 2011); giving `vmin`/`vmax` (or `metric volume`) turns this on and adds a volume output column |

Both windows default to "no bound" (`rmin 0`, `rmax +∞`, likewise for volume). All four
thresholds may be **equal-style variables** (e.g. `rmin v_myrmin`) that are re-evaluated
every block, so the window can be time-dependent.

The clearance window is how you pick a *sublattice by geometry*. In fcc with lattice
constant *a*, for instance:

| Interstitial | Sites / cell | Clearance |
|---|---|---|
| Octahedral | 4 | *a*/2  ≈ 1.76 Å at *a* = 3.52 |
| Tetrahedral | 8 | √3·*a*/4  ≈ 1.52 Å at *a* = 3.52 |

So `rmin 1.6 rmax 2.0` keeps **only octahedral** sites; a lower `rmin` (e.g. `1.4`) admits
the tetrahedral set as well. (This is exactly the `in.mc_sites.langmuir` example.)

```lammps
compute OCT all sites/voronoi rmerge 0.3 rmin 1.6 rmax 2.0                 # octahedral only
compute BIG all sites/voronoi rmerge 0.3 vmin 8.0 vmax 20.0 metric volume  # by probe volume
```

### 2. Restrict to a region of the box

`region` limits the sites geographically — charge only a surface slab, only the atoms near
a grain boundary, only one phase, etc. It exists on **both** commands and they do different
things:

| Command | `region` keyword effect |
|---|---|
| `compute sites/voronoi region <rID>` | only *empty candidate sites* inside the region are produced (evaluated at build time) |
| `fix mc/sites region <rID>` | restricts both the sites used **and** the existing species atoms eligible for deletion to the region |

For a self-consistent GCMC restricted to a sub-volume, put the same region on both:

```lammps
region  slab block INF INF INF INF 10.0 30.0 units box
compute SITES all sites/voronoi rmerge 0.3 rmin 1.6 region slab
fix     MC    all mc/sites 100 200 2 12345 500.0 sites c_SITES mode gc mu -2.4 region slab
```

### 3. Coordination filter — surfaces, voids, subsurface (`coord`)

Voronoi vertices generated in **vacuum** (outside a free surface, or in the padding of a
non-periodic box) have large clearance but few nearby atoms — they are not real
interstitial sites. `coord k rcut` keeps a site only if **at least `k` host atoms lie
within `rcut`** of it, which cleanly separates buried/interstitial sites from vacuum
vertices. By default the counted atoms are the compute group; use `hostgroup` to count a
specific species (e.g. only the metal, ignoring already-inserted H).

| Keyword | Meaning |
|---|---|
| `coord <k> <rcut>` | require ≥ `k` atoms within `rcut` (`k ≥ 1`, `rcut > 0`) |
| `hostgroup <group>` | which atoms the `coord` count uses (default: the compute group) |

```lammps
# fcc slab with free surfaces in z: drop the vacuum vertices, keep only
# sites with a full first shell of metal atoms around them
group   metal type 1
compute SITES all sites/voronoi rmerge 0.3 rmin 1.4 coord 6 3.5 hostgroup metal
```

Raising `k` biases toward fully-embedded (bulk-like) sites; a modest `k` (≈ 4–6 for fcc
first shell) is enough to remove surface/vacuum artifacts while keeping true subsurface
interstitials.

### 4. Occupancy veto — don't propose on top of an existing atom (`occcut`)

In a **dynamic** run the tessellation can still place a candidate vertex essentially where
one of your Monte Carlo species atoms already sits. `occcut dist exclgroup <group>` drops
any site within `dist` of an atom of `exclgroup` (the two keywords are required together).
Point `exclgroup` at your inserted species so the compute only ever offers *genuinely
empty* sites; the fix separately re-adds the occupied species atoms as deletable entries,
so nothing is double-counted.

```lammps
group   hyd type 2
compute SITES all sites/voronoi rmerge 0.3 rmin 1.4 occcut 0.9 exclgroup hyd
```

(The static `file` site source doesn't need this — a file site holding a species atom is
recognized as occupied by exact position match. `occcut` is the dynamic-catalogue
equivalent.)

### 5. Everything at once

The criteria are independent and order doesn't matter; a real run typically stacks several:

```lammps
compute BULK all sites/voronoi rmerge 0.3 rmin 1.4 rmax 2.5 metric volume vmin 8.0 vmax 20.0 region slab coord 6 3.5 hostgroup metal occcut 0.9 exclgroup hyd
```

That single site definition keeps a vertex only if it is octahedral-ish by clearance
(`rmin`/`rmax`) **and** within the probe-volume window (`vmin`/`vmax`) **and** inside
`slab` (`region`) **and** embedded with ≥ 6 metal neighbors (`coord`/`hostgroup`, no
surface or vacuum vertices) **and** not already occupied by an inserted atom
(`occcut`/`exclgroup`). LAMMPS input lines may be wrapped with a trailing `&` if you prefer
one keyword per line.

The compute writes one row per surviving site: `x y z clearance [volume] coord` (6 columns
when a volume window/metric is active, 5 otherwise; the `coord` column is 0 when `coord` is
unused), plus a global scalar = total number of sites. You can drive the fix with it, or
just `dump local` it to analyze the interstitial-site distribution without any MC.

### 6. The fix side — how insertion/deletion behaves

Once the sites are chosen, `fix mc/sites` controls the sampling:

| Keyword / arg | Meaning |
|---|---|
| `Nevery Ntrials type seed Temp` | MC block every `Nevery` steps; `Ntrials` trial flips per block; species atom `type`; RNG `seed`; MC temperature `Temp` (may be `v_...`) |
| `sites c_ID` \| `sites file <path>` | dynamic catalogue from a compute, or a static site list (validation) |
| `mode gc mu <value>` | grand-canonical insert/delete; `mu` is the lattice-gas chemical potential (may be `v_...`) |
| `mode source rate <n>` | irradiation-style: `n` unconditional insertions per block, no acceptance test |
| `region <rID>` | restrict sites and deletable atoms to a region (see §2) |
| `overlap_cutoff <d>` | reject an insertion whose site clearance < `d` *without* an energy evaluation (default 0 = off; needs a compute catalogue) |
| `maxspecies <N>` | stop inserting once there are `N` species atoms (upper occupancy cap) |
| `charge <q>` | charge given to inserted atoms (needs a charge atom style) |
| `tfac_insert <f>` | scale factor on the inserted-atom velocity temperature (default 1.0) |

Acceptance is the symmetric lattice-gas rule
`P_ins = min(1, e^(−β(ΔU − μ)))`, `P_del = min(1, e^(−β(ΔU + μ)))`, with ΔU a **full**
energy evaluation (as verified in the source: `fix_mc_sites.cpp`, `attempt_insertion` /
`attempt_deletion`). Deletion genuinely removes the atom for the trial and restores it
exactly on rejection, which is more correct than `fix gcmc`'s exclusion-group masking for
many-body/ML potentials.

**Output vector** (length 8, e.g. `f_MC[6]`): (1) trial attempts, (2) accepted insertions,
(3) accepted deletions, (4) current species count *N*, (5) catalogue size *M*, (6)
instantaneous site concentration *N*/*M*, (7) overall acceptance ratio, (8) skipped
source-mode insertions.

Full syntax, defaults, and restrictions are in the two manual pages —
[`doc/compute_sites_voronoi.rst`](doc/compute_sites_voronoi.rst) and
[`doc/fix_mc_sites.rst`](doc/fix_mc_sites.rst) — which render in the LAMMPS HTML manual once
the tree is built. The runnable [`examples/`](examples/) (with a
[walkthrough](examples/README.md)) show the octahedral Langmuir check and a hybrid MD/MC run.

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
