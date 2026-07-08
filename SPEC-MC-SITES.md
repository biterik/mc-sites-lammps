# SPEC — `compute sites/voronoi` + `fix mc/sites`

> Implementation spec for Claude Code. Design finalized 2026-07-07 by Erik Bitzek (with
> Claude); decisions in §3 are settled — record objections in `QUESTIONS.md`, do not
> redesign. Parent context: `../MC-DRIVER/` (validated Python reference, spec, design
> notes `docs/DESIGN-NOTES.md`, prior source-analysis brief `docs/LAMMPS-SITE-GCMC-BRIEF.md`).

---

## 1. Purpose

Native LAMMPS capability for **occupancy Monte Carlo of interstitial species on a
dynamic, geometry-derived site catalogue**, in hybrid MD/MC:

- Sites = **vertices of the Voronoi tessellation of all atoms** (centers of the largest
  empty spheres). Works for perfect crystals, defects, voids/bubbles, surfaces — no
  assumed sublattice.
- The catalogue is **recomputed at every MC invocation** (after the MD steps in between),
  filtered by user criteria (clearance/volume window, region, coordination) that may
  themselves be time-dependent.
- MC moves: **lattice-gas grand-canonical** insertion/deletion (mode *gc*) or
  **fixed-rate insertion** (mode *source*, irradiation-style). VC-SGC is v2 (§12).
- Target applications: H in Ni (reference case, MC-DRIVER), H₂ formation, He bubbles,
  O/H₂O with suitable potentials. Species-agnostic, potential-agnostic.

Goal 1: works in Erik's GRACE-enabled fork build. Goal 2: upstreamable to LAMMPS
(`develop`) per `LAMMPS-contributing-guide.md`.

## 2. Deliverables

| # | Item | Location (in LAMMPS tree) |
|---|------|--------------------------|
| 1 | `compute sites/voronoi` | `src/VORONOI/compute_sites_voronoi.{h,cpp}`, class `ComputeSitesVoronoi` |
| 2 | `fix mc/sites` | `src/MC/fix_mc_sites.{h,cpp}`, class `FixMCSites` |
| 3 | Docs | `doc/src/compute_sites_voronoi.rst`, `doc/src/fix_mc_sites.rst` + command-table registration |
| 4 | Examples | `examples/PACKAGES/mc_sites/` (`in.mc_sites.langmuir`, `in.mc_sites.dynamic`, small + fast, guide §6 rules) |
| 5 | Tests | project-level test scripts under `tests/` in THIS folder (LAMMPS inputs + python analysis), see §8 |
| 6 | `ISSUE-DRAFT.md` | this folder — the upstream GitHub issue text (feature proposal, guide §2) |
| 7 | Phase-6 cluster scripts (proposed, not run) | this folder, `cluster/` |
| 8 | `SITE_GCMC_ANALYSIS.md` | this folder — Phase 0 report |

Naming is fixed (Erik's choice): `sites/voronoi`, `mc/sites`. Header guards, FixStyle/
ComputeStyle macros, file names follow mechanically per the contributing guide.

## 3. Settled design decisions — do not reopen

1. **Representation: real create/delete** of atoms of the MC species (no ghost/vacancy
   type, no type swaps).
2. **Empty-site discovery: tessellation of ALL atoms** (host + MC species + anything
   else). A site's clearance (empty-sphere radius) is its distance to the nearest atoms —
   the insertion clearance by construction.
3. **Occupancy: the MC-species atoms ARE the occupied sites** — their current positions.
   No proximity matching, no occupancy tolerance, no orphan handling. Deletable = atoms of
   the fix's species type in the fix group and inside the region (if given).
4. **Frozen catalogue per MC block.** At each fix invocation: build catalogue = empty
   entries (from the compute or a static file) + occupied entries (current species atoms).
   Freeze it. All trials in this block use only the frozen catalogue. A deletion leaves
   its entry in the catalogue as *empty at the deleted atom's position* (reverse move
   exists within the block); an accepted insertion marks its entry occupied.
5. **Move: symmetric single-entry flip.** Pick one catalogue entry uniformly at random;
   empty → propose insertion at its position; occupied → propose deletion of that atom.
   No proposal-ratio corrections.
6. **Acceptance: lattice-gas convention, own μ reference** (resolved in
   `../MC-DRIVER/docs/DESIGN-NOTES.md` §7):
   insertion `min(1, exp(-β(ΔU - μ)))`, deletion `min(1, exp(-β(ΔU + μ)))`,
   β = 1/(k_B T). NO ideal-gas factors (`zz`, volume, 1/(N+1)) — dropped, not converted.
   μ differs from fix gcmc's scale by a constant, calibrated externally when needed.
7. **ΔU via full-energy evaluation** (fix gcmc `energy_full()` pattern): exchange +
   reneighbor + `pair->compute` with `eflag = ENERGY_GLOBAL | ENERGY_ONLY`. Required for
   EAM/MEAM/ACE/GRACE/hybrid anyway. Design the energy-evaluation call behind a small
   internal interface so a local-ΔU backend can be added later without restructuring (§12).
8. **Detailed balance stance:** exact within a block (frozen catalogue); the block-to-block
   rebuild is a documented approximation, quantified by the rebuild-invariance test (§8.4).
9. **Multi-species by stacking:** the fix is single-species; multiple instances (one per
   species, each with own type, μ, criteria/compute) coexist. Guard against two fixes
   claiming the same species type (error).
10. **Modes in v1: `gc` and `source`** (accept-all insertion of a prescribed number of
    atoms per invocation at randomly chosen empty catalogue sites). VC-SGC deferred to v2.
11. **Inserted-atom velocities:** Maxwell-Boltzmann at T, with a `tfac_insert`-style
    scale factor (fix gcmc pattern). `charge` keyword assigns q to inserted atoms when the
    atom style has charge.
12. **Site metrics:** empty-sphere radius always computed (free from voro++);
    **probe Voronoi volume** available as an optional, more expensive metric
    (insert probe point, local re-tessellation, read its cell volume). The filter window
    can reference either (`rmin/rmax` vs `vmin/vmax`). Radius is the computational
    default; volume matches the von-Pezold E^H(V) surrogate convention.
13. **Vertex merging:** vertices closer than `rmerge` are clustered (single-linkage,
    replaced by their centroid; clearance = min over members). Required: thermal noise
    splits degenerate ideal-lattice vertices.
14. **Criteria in v1:** clearance/volume window + `region` + **coordination** (≥ k atoms
    of a host group within r_c of the site — distinguishes internal voids from outer
    vacuum and controls surface-site participation). Thresholds accept equal-style
    variables where cheap to support; region membership is evaluated at rebuild time only.

## 4. Codebase & git strategy

- Clone **`https://github.com/thermoatoms/lammps`**, branch `develop`, into `./lammps/`.
  Record the HEAD commit in `PROGRESS.md`. (Context: base is upstream `patch_11Feb2026` +
  GRACE pair styles + MC extensions; see `../LAMMPS062025_update4/FORK-ANALYSIS.md`.)
- Create a local branch `feature/mc-sites`. Commit per phase. **Never push.**
- New files only, except these registration edits (allowed): the VORONOI and MC package
  file lists in `cmake/CMakeLists.txt` / package `Install.sh` if style headers are not
  auto-discovered (check first — most styles ARE auto-discovered via `*_style` headers),
  and doc command tables (`doc/src/Commands_*.rst`, `doc/src/fix.rst`/`compute.rst` lists).
- **Upstream compatibility rule:** use no API that differs between the fork and current
  upstream `develop`. Phase 0 checks the few risk spots (`Pair::extract`, eflag constants,
  `create_atom` signatures) against upstream via `https://raw.githubusercontent.com/lammps/lammps/develop/...`.
- Local build (macOS, no GRACE/TF needed):
  `cmake -B build -D PKG_MC=on -D PKG_VORONOI=on -D DOWNLOAD_VORO=on [-D BUILD_MPI=on]`
  then `cmake --build build -j`. Also build once with `-D LAMMPS_SIZES=bigbig` (gate §9).

## 5. `compute sites/voronoi`

**Syntax (proposed; refine details in Phase 0, keep names):**

```
compute ID group-ID sites/voronoi rmerge 0.3 rmin 1.4 rmax 2.5 &
    region rID coord 6 3.5 hostgroup metal metric radius occcut 0.9
```

- `group-ID`: atoms whose tessellation defines the sites (normally `all`).
- Keywords: `rmerge <dist>`; `rmin/rmax <dist>` (clearance window); `vmin/vmax <vol>`
  (probe-volume window ⇒ activates probe metric); `region <rID>`; `coord <k> <rcut>`
  with `hostgroup <gID>`; `occcut <dist>` — drop candidate sites within `occcut` of an
  atom of a group given by `exclgroup <gID>` (lets the fix exclude sites essentially on
  top of existing species atoms; default off).
- **Output:** local array (`array_local`): one row per accepted site, columns
  `x y z clearance [volume] coord`; plus global scalar = number of sites. Follow an
  existing local-array compute as template (e.g. `compute pair/local` or the local output
  of `compute voronoi/atom`).

**Algorithm:** per-rank voro++ cell computation for owned atoms (mirror
`src/VORONOI/compute_voronoi_atom.cpp`'s container setup incl. ghost handling and
triclinic); extract each cell's vertices + vertex radii (voro++ `cell.vertices()` /
vertex distance = |vertex − generator|); collect vertices in the rank's subdomain
(dedupe across cells: a vertex is shared by neighboring cells — merge via `rmerge`
clustering, keep subdomain ownership unambiguous for vertices near rank boundaries:
owner = rank whose subdomain contains the merged centroid); apply criteria; store.
Non-periodic dims: voro++ clamps cells at the container — vacuum vertices get large
clearance; the coordination criterion is the intended discriminator (document this).

**Cost target:** O(N) per rebuild; no global gathers except the site count (the fix does
the gather, §6). Do not optimize beyond straightforward correctness in v1.

## 6. `fix mc/sites`

**Syntax (proposed):**

```
fix ID group-ID mc/sites Nevery Ntrials Ttype seed Temp &
    sites c_SITES|file sites.txt  mode gc mu -2.40 | mode source rate 5 &
    region rID  tfac_insert 1.0  charge 0.0  maxspecies N
```

- `Nevery`: MD steps between MC blocks. `Ntrials`: trial flips per block. `Ttype`: atom
  type (or type label) of the MC species. `Temp`: MC temperature (may be an equal-style
  variable, like the fork's fix atom/swap supports).
- `sites c_ID`: consume the compute's local array (dynamic catalogue — the fix triggers
  the compute at each invocation). `sites file <path>`: static site list (validation mode;
  columns x y z, one site per line; positions in box units, wrapped).
- `mode gc mu <μ>` or `mode source rate <n_insert_per_invocation>`.
- Inserted atoms: assigned to `group-ID` (and `all`), type `Ttype`, MB velocities at
  `Temp*tfac_insert`, charge per keyword.

**Per-invocation algorithm (pre_exchange, following fix gcmc's placement in the timestep):**

1. If dynamic: trigger the sites compute; gather all ranks' sites into a global catalogue
   (positions + metadata), consistent on all ranks (`MPI_Allgatherv`; the catalogue is
   small — thousands of entries).
2. Append occupied entries: global list of (tag, position) of species atoms in
   group∩region. (Collect like fix gcmc's `update_gas_atoms_list()` /
   `ngas_before` bookkeeping — see `fix_gcmc.h`.)
3. Freeze. For `Ntrials` iterations: draw entry index with the **all-ranks-synchronized
   RNG** (`random_equal`, `RanPark` — fix gcmc pattern); empty → insertion attempt at that
   position, occupied → deletion attempt of that tag; evaluate ΔU = E_after − E_before via
   full energy (§3.7); accept with §3.6; on accept update catalogue entry + atom system,
   on reject restore exactly (fix gcmc's insert/delete/restore mechanics — Phase 0 maps
   the exact calls: `avec->create_atom`, tag assignment/`tag_extend`, `atom->natoms`
   updates, `atom->map_init/map_set`, deletion via copy-last + `nlocal--`, borders/comm).
4. `source` mode: choose `rate` distinct empty entries (synced RNG), insert
   unconditionally (still respect an overlap guard: skip if clearance criterion violated —
   count skips in the output vector).
5. Bookkeeping: `energy_stored` chaining like fix gcmc to avoid one redundant full-energy
   call per trial; **overlap pre-rejection**: for insertions, if the entry's clearance
   `< overlap_cutoff` (keyword, default 0 = off) reject without any energy call.
6. **Energy-only hint:** set `eflag = ENERGY_GLOBAL | ENERGY_ONLY` (upstream idiom —
   `fix.cpp:201`, `pair.h eflag_only`; no upstream pair honors it, harmless). Additionally,
   if `pair->extract("compute_energy_only", dim)` returns non-null (the fork's GRACE
   styles: `pair_grace.cpp`, `PairGRACE::extract`), set that int flag to 1 for the
   duration of MC energy evaluations and restore it after — this activates GRACE's
   energy-only TF signature. Zero core-file changes; degrades gracefully on upstream.

**Output** (`compute_vector`, intensive/extensive flags per guide): attempts, accepted
insertions, accepted deletions, current N_species, catalogue size M (empty+occupied),
instantaneous c = N_occupied/M, acceptance ratio, source-mode skips.

**Restart:** follow fix gcmc's `write_restart`/`restart` for RNG state + counters. The
catalogue itself is derived state — nothing to save.

**MPI correctness (v1 scope):** proposals identical on all ranks via `random_equal`;
create/delete executed by the owning rank (position ∈ subdomain — fix gcmc pattern);
full-energy evaluation is collective. This is gcmc-grade parallelism (serial in moves,
parallel in energy). Sadigh-Erhart checkerboarding: v2, out of scope.

**Errors/warnings:** require 3D, reneighboring enabled (`neigh_modify once yes` → error,
as fix gcmc docs demand); require atom map for GRACE-style potentials (init it like
`pair_grace.cpp init_style` does if absent); warn if acceptance ratio leaves [0.01, 0.99]
persistently; error if two mc/sites fixes share a species type; error if `Ttype` has no
mass; specific `{fmt}`-style messages per guide §5.

## 7. Compatibility & restrictions (verify in Phase 0; document in .rst)

- Works with any potential providing correct total energies via `pair->compute`:
  pairwise, EAM, MEAM, ACE/PACE, GRACE (fork), hybrid. kspace included in the full-energy
  path if present (expensive; net-charge insertion caveats documented).
- MEAM: known sensitivity to repeated create/delete (MC-DRIVER SPEC §12) — the round-trip
  guard (§8.3) is the canary; run it with MEAM in Phase 6 (cluster) since no MEAM
  potential file is available locally.
- GRACE (fork): padding absorbs ±N changes (`pair_grace.cpp` GracePaddingDimension);
  occasional TF graph recompiles are expected and amortized; `newton on` + full neighbor
  list + atom map required. Recommend a nonzero `padding` fraction in the doc.
- NOT supported v1 (document as Restrictions): Kokkos/GPU-package/INTEL suffix styles
  (no MC fix is Kokkos-ported — same status as fix gcmc), ReaxFF/charge-equilibration,
  core-shell, molecule-template insertion, 2D.
- GPU note for docs: GRACE's GPU execution is internal to TensorFlow; the fix works
  unmodified with TF-GPU GRACE builds.

## 8. Validation suite (the definition of correct)

Analysis scripts in python (numpy) under `tests/`; every test records seed; tolerances
explicit. `../MC-DRIVER` is the oracle — reuse its analytic targets and, where present,
its cross-check test (`tests/test_lammps_native_crosscheck.py`; its LAMMPS halves were
written for fix gcmc/sgcmc and are skip-guarded — adapt copies for fix mc/sites here,
do not modify MC-DRIVER).

1. **Compute unit test (geometry).** Ideal fcc, a = 3.52 Å, pair_style zero, no MD.
   Analytic: octahedral sites (4/cell) have clearance a/2 = 1.760 Å; tetrahedral (8/cell)
   have √3·a/4 ≈ 1.524 Å. With `rmerge` ≈ 0.3 Å the compute must find exactly 4+8 merged
   sites/cell; window `rmin 1.6` selects exactly the 4 octahedral. Perturb atoms by
   0.05 Å (fixed seed): same counts (merging works). Triclinic variant of the same box:
   same counts. Surface slab + `coord` criterion: no vacuum sites; void variant (remove an
   atom cluster): interior sites found, counts recorded against a reference run.
2. **Langmuir isotherm (fix, static file sites, pair zero).** ε = 0 ⇒
   θ(μ,T) = 1/(1+exp(−μ/k_BT)). Sweep ≥ 6 μ values at T = 300 K and 600 K on a ≥200-site
   list; target |θ_MC − θ_analytic| < 0.02 per point. Extreme-μ guards: μ→±large ⇒ θ→1/0.
3. **Round-trip guard (fix, pair zero + lj/cut).** Insert-then-delete same site returns
   total energy to baseline ≤ 1e-9 eV and N to baseline; reject path leaves state
   bit-identical (energy re-check); mirrors MC-DRIVER PROGRESS "round-trip guards".
4. **Rebuild-invariance (the DB quantification).** Static frozen host (no integration
   fix), dynamic `sites c_ID` catalogue: rebuilding every block must reproduce the static
   file-list isotherm within MC error — rebuild changes nothing on a static lattice.
5. **Enumeration cross-check (interacting).** Small site list (≤12) with `lj/cut`
   interactions between inserted atoms (frozen host, pair zero for host via hybrid or a
   host-free box): exact ⟨c⟩ and P(N) by direct 2^M enumeration with the same LAMMPS
   energies (script may call LAMMPS once per configuration); MC must match P(N) within
   error. Adapt MC-DRIVER's enumeration methodology.
6. **Hybrid MD/MC smoke test.** lj/cut host at low T + NVT integration between blocks +
   dynamic catalogue: runs stably, energy conserving between MC blocks, c(μ) monotone in μ
   (3 μ points).
7. **Source mode:** exact insertion counts per invocation; atoms appear only at
   criteria-passing sites; skip counter works when the catalogue is exhausted.
8. **Stacking:** two fix instances (two species types) coexist; per-species counters
   correct; error path for duplicate species type triggers.
9. **Parallel/portability gates:** serial vs `mpirun -np {2,4}` give statistically
   consistent isotherm points (and identical results where determinism is expected, e.g.
   source-mode placement with fixed seed); SMALLBIG and BIGBIG both compile + pass test 2.

## 9. Phases & gates

0. **Source analysis (read-only in the LAMMPS tree)** → `SITE_GCMC_ANALYSIS.md`.
   Update `../MC-DRIVER/docs/LAMMPS-SITE-GCMC-BRIEF.md`'s questions to this design (that
   brief predates the dynamic-catalogue/compute+fix decisions — where they conflict, THIS
   spec wins). Map every mechanism the fix needs to `file:function` in the actual
   checkout: gcmc insertion/deletion/restore, `energy_full()` (`fix_gcmc.cpp:2325` at
   stable), RNG sync, gas-list bookkeeping, `compute_voronoi_atom` container setup, voro++
   vertex API, local-array compute template, fork `Pair::extract` hook
   (`pair_grace.cpp`), fork-vs-upstream drift on used APIs. Gate: report complete, plan
   confirmed or amended in `QUESTIONS.md`.
1. **`compute sites/voronoi`** + test §8.1. Gate: 8.1 green (serial; MPI if available).
2. **`fix mc/sites` gc mode, static file sites** + tests §8.2, §8.3. Gate: green.
3. **Dynamic coupling** (compute-driven catalogue) + tests §8.4, §8.5, §8.6. Gate: green.
4. **source mode, stacking, keywords** (`charge`, `tfac_insert`, overlap) + §8.7, §8.8.
5. **Compliance & packaging:** `.rst` docs (build `make html` if sphinx available, else
   lint by inspection against guide §4), examples, clang-format, BIGBIG build, MPI runs
   (§8.9), `ISSUE-DRAFT.md` (feature proposal: motivation incl. missing-capability
   analysis vs fix gcmc/sgcmc/atom-swap, design summary, validation evidence). Gate:
   checklist in the contributing guide §8 all ticked (CI items marked "pending PR").
6. **Cluster handoff (propose only):** scripts + instructions for Erik: fork build with
   the new files (adapt `../LAMMPS062025_update4/mpcdf-lammps/build-lammps-*-fork.sh`),
   MEAM round-trip canary, EAM Korbmacher anchor run (μ sweep, compare MC-DRIVER
   PROGRESS step-8 targets: plateau μ_M = −2.405 eV, a(c) 3.520→3.738 Å), GRACE smoke
   test with energy-only flag on/off timing comparison. SLURM only, PTMP paths, per
   `../../CLAUDE.md` cluster policy (quoted in the scripts' headers).

## 10. Style & compliance

Everything in `LAMMPS-contributing-guide.md` applies from the first line of code: GPL-2.0
header + author (Erik Bitzek, erik.bitzek@googlemail.com) on every new file; style-header
macro wrapped in `// clang-format off/on`; base-class-only includes in style headers;
pointer members nullptr-initialized in constructors; `static constexpr` constants;
rank-0 output via `utils::logmesg`; specific error messages; ASCII-only American-English
.rst; small fast examples with the naming rules of guide §6.

## 11. Known facts & code anchors (verified 2026-07-07, re-verify in checkout)

- Fork: `thermoatoms/lammps` @ `develop`; version "11 Feb 2026 / MCnoforce-localE"; GRACE
  pair styles in `src/ML-PACE/`; fork's `fix atom/swap` has `noforce`/`localE` (PACE-only)
  — reference, not a dependency.
- `ENERGY_ONLY` eflag exists upstream; parsed into `eflag_only` by base classes
  (`pair.cpp:963`, `fix.cpp:217` at stable); **no upstream pair style honors it** (grepped).
- GRACE energy-only hook: `PairGRACE::extract("compute_energy_only")` returns
  `&flag_compute_energy_only`; energy-only TF signature used when the model provides it,
  warns-once otherwise (`pair_grace.cpp`).
- GRACE padding: `GracePaddingDimension` for atoms + neighbors; graph recompiles on
  resize; `pair_forces` mode auto-forced when nprocs > 1; requires newton on, full
  neighbor list, atom map.
- fix gcmc doc: full_energy forced for EAM/manybody/hybrid/kspace/tail/fix-energies;
  reneighboring required; 3D only; MC part scales poorly — all applies to us, cite
  analogously in our docs.
- fcc geometry for §8.1: oct clearance a/2, tet clearance √3a/4; ideal-lattice Voronoi
  vertices are degenerate (rhombic dodecahedron) — merging is mandatory even at 0 K
  (floating-point splitting).
- Benchmarks for cost statements (Erik's clusters, `mpcdf-lammps/STATUS.md`): GRACE-2L
  ≈ 0.94 katom·step/s, 1L ≈ 4.33 (viper MI300A); EAM/PACE orders of magnitude cheaper.
  Full-energy per trial is THE cost for GRACE — motivates the energy-only flag now and
  local-ΔU later.

## 12. Out of scope / v2 (do not implement; do not preclude)

- **VC-SGC mode** (constraint Φ = U − Δμ·N + κ̄·N_ref·(c−c₀)²; open design point: c₀
  references instantaneous catalogue size M vs fixed N_ref — Erik decides in v2).
- **Local-ΔU backend** (receptive-field evaluation; fork localE generalization). Keep the
  energy call behind one internal method so this can slot in.
- Kokkos port; Sadigh-Erhart parallel checkerboard MC; molecule insertion; transmutation
  (species-swap) moves; per-site proposal weights (surrogate-based importance sampling,
  DESIGN-NOTES Idea B).
