# ISSUE-DRAFT.md — upstream feature proposal (to be filed by Erik on github.com/lammps/lammps)

> Per the contributing guide §2 ("talk to the developers before writing anything
> non-trivial" — here the implementation exists and is validated, so this issue both
> proposes the feature and offers the finished contribution). File as a GitHub issue of
> type "feature request / contribution offer". Do not submit a PR before developer
> feedback on this issue.

---

**Title:** [Feature/contribution] Occupancy Monte Carlo on a dynamic, Voronoi-derived
interstitial-site catalogue: `compute sites/voronoi` (VORONOI) + `fix mc/sites` (MC)

## Summary

I would like to contribute two coupled styles, developed and validated against the
current `develop`-based tree:

- **`compute sites/voronoi`** (VORONOI package): builds a catalogue of interstitial
  sites as the merged vertices of the Voronoi tessellation of the current atomic
  configuration — the centers of locally largest empty spheres, with their clearance
  (empty-sphere radius) and optionally their probe Voronoi volume — filtered by
  clearance/volume windows, a region, a coordination criterion, and an occupied-site
  veto. Local array output (one row per site) + global scalar (site count).
- **`fix mc/sites`** (MC package): lattice-gas grand-canonical MC (and a fixed-rate
  "source" insertion mode) of a single species on that catalogue, in hybrid MD/MC:
  the catalogue is rebuilt at every MC invocation (so it follows the evolving
  microstructure), frozen for the block, and sampled with symmetric single-entry flips
  (empty entry → insertion at its position, occupied entry → deletion of that atom),
  with full-energy Delta-U evaluation following the `fix gcmc` full_energy pattern.

## Missing capability (why the existing MC fixes do not cover this)

| Existing | Limitation for interstitial absorption/segregation problems |
|---|---|
| `fix gcmc` | Inserts at uniform random positions in a region: in a condensed phase, virtually all insertion trials land in repulsive cores and are rejected; the ideal-gas reservoir reference is unnecessary for lattice-gas occupancy problems. No notion of geometry-derived sites. |
| `fix sgcmc` (VC-SGC) | Transmutation-only (type swaps on existing atoms): requires pre-seeded placeholder ("ghost") atoms on every candidate site; no true insertion/deletion; ghost species are ill-defined for EAM/ML potentials; site list is static by construction. |
| `fix atom/swap` | Same transmutation restriction; no atom-count changes. |
| `fix widom` | Test insertions only (no state changes). |

The proposed styles target problems like H in Ni/steels (absorption isotherms,
hydride formation), He bubble growth in irradiated metals, and O/H2O uptake — where
the species occupies interstitial cavities whose positions *move and change* with the
defect microstructure (dislocations, GBs, voids, surfaces). The Voronoi-vertex
construction requires no assumed sublattice: it discovers octahedral/tetrahedral
sites in crystals, adapts continuously near defects, and finds cavity sites in voids
and bubbles. Proposing only at geometric cavities raises MC acceptance by orders of
magnitude compared to random placement, which matters enormously when Delta-U costs a
full energy evaluation of a many-body/ML potential.

## Design summary

- Sites = vertices of the voro++ tessellation of ALL atoms, computed per-processor
  exactly like `compute voronoi/atom` (subdomain + ghost-cutoff container, ghosts
  included); vertices merged by single-linkage clustering (`rmerge`, required —
  ideal-lattice vertices are degenerate and split by fp noise); per-rank ownership by
  the merged centroid's subdomain with a deterministic tie-break for sites exactly on
  subdomain/periodic faces (validated: serial and MPI produce identical catalogues).
- The species atoms ARE the occupied sites (their current positions); no proximity
  matching or orphan handling. Frozen catalogue per MC block; deletion leaves an
  empty entry at the deleted position (reverse move exists within the block).
  Detailed balance is exact within a block; the block-to-block rebuild is a
  documented approximation, quantified by a rebuild-invariance test (rebuild changes
  nothing on a static lattice — verified).
- Acceptance: lattice-gas convention min(1, exp(-beta(dU -/+ mu))), own mu reference
  (the ideal-gas factors are dropped, not converted; documented).
- Move mechanics transplant `fix gcmc`'s battle-tested patterns: synchronized RanPark
  proposal + acceptance streams on all ranks, owner-rank create/delete, collective
  full-energy evaluation via the thermo_pe compute. One deliberate improvement:
  deletion trials REALLY remove the atom (pack_exchange snapshot, exact
  unpack_exchange restore on reject) instead of `fix gcmc`'s exclusion-group masking,
  which leaves the isolated-atom energy E0 in the total — E0 = 0 for EAM/LJ, but
  nonzero for ACE/GRACE-type ML potentials, where masking would bias every deletion.
- New files only: `src/VORONOI/compute_sites_voronoi.{h,cpp}`,
  `src/MC/fix_mc_sites.{h,cpp}`, two doc pages, one example dir. Zero core changes;
  both packages auto-glob sources (only doc tables and src/.gitignore touched).
- The full-energy call sets `eflag = ENERGY_GLOBAL | ENERGY_ONLY` (the upstream hint;
  no upstream pair style honors it today, harmless) and is isolated behind one
  internal method so a local-Delta-U backend can slot in later without restructuring.

## Validation evidence (all automated, tolerance-explicit, seeds recorded)

1. **Geometry unit test**: ideal fcc (a=3.52): exactly 12 merged sites/cell; oct
   clearance = a/2 and tet = sqrt(3)a/4 to <1e-6; window rmin 1.6 selects exactly the
   4 oct/cell; counts invariant under 0.05 A random perturbation and under a
   triclinic (prism) representation; slab + coordination criterion yields no vacuum
   sites; void test finds interior cavity sites. Serial == np2 == np4 catalogues.
2. **Langmuir isotherm** (216 file sites, pair zero): 9 (mu,T) points, max
   |theta_MC - theta_exact| = 0.004 (tolerance 0.02); extreme-mu limits reach 0/1.
3. **Round-trip guard** (pair zero AND lj/cut): fill-all-sites then empty returns the
   total energy to baseline < 1e-9 eV and N exactly; insertion- and deletion-reject
   paths leave N and E unchanged; trajectories bit-reproducible from the seed.
4. **Rebuild-invariance**: dynamic (rebuilt-every-block) catalogue reproduces the
   static-file isotherm and the analytic curve within 0.006 on a frozen lattice;
   catalogue size exactly conserved.
5. **Exact enumeration** (12-site ring, lj/cut couplings = NN lattice gas with
   J = -0.05 eV): MC matches the 2^12 partition-function reference: <c> error 0.002,
   P(N) total-variation distance 0.01.
6. **Hybrid MD/MC**: lj/cut host + NVT + dynamic catalogue runs stably; c(mu)
   monotone.
7. **Source mode**: exact per-invocation insertion counts, placement only at
   criteria-passing sites, skip counter on catalogue exhaustion.
8. **Stacking**: two instances (two species types) coexist with independent
   isotherms; duplicate-species-type error path verified.
9. **Portability**: SMALLBIG and BIGBIG builds both compile and pass the Langmuir
   test; serial vs mpirun -np {2,4} isotherms consistent; source-mode placement
   bit-deterministic across decompositions.

## Questions for the developers

1. Naming: `compute sites/voronoi` + `fix mc/sites` acceptable? (`fix gcmc/sites`
   was avoided since the acceptance is deliberately NOT gcmc's ideal-gas form.)
2. Package placement: compute in VORONOI (voro++ dependency) and fix in MC — or
   should both live in one package to keep them coupled?
3. Is the lattice-gas mu reference (documented as differing from fix gcmc's by a
   calibratable constant) acceptable, or is an optional ideal-gas-compatible mode
   wanted?
4. Any interest in the follow-ups we have designed for: VC-SGC mode on the same
   catalogue, and a local-Delta-U backend behind the existing energy-evaluation
   seam?

I can open a PR with the complete contribution (code, .rst docs, examples) rebased on
current `develop` as soon as the design points above are settled.
