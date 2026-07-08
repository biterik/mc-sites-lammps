.. index:: fix mc/sites

fix mc/sites command
====================

Syntax
""""""

.. code-block:: LAMMPS

   fix ID group-ID mc/sites Nevery Ntrials type seed Temp sites source mode modeargs keyword values ...

* ID, group-ID are documented in :doc:`fix <fix>` command
* mc/sites = style name of this fix command
* Nevery = invoke a Monte Carlo block every this many steps
* Ntrials = number of trial flips per block (*gc* mode)
* type = atom type (or type label) of the Monte Carlo species
* seed = random # seed (positive integer)
* Temp = Monte Carlo temperature (temperature units, may be an equal-style variable)
* sites source = *c_ID* or *file path*

  .. parsed-literal::

       *c_ID* = local array of a sites-producing compute (dynamic catalogue,
                e.g. :doc:`compute sites/voronoi <compute_sites_voronoi>`)
       *file path* = static site list, one "x y z" triple per line
                     (validation mode; positions in box distance units)

* mode = *gc* or *source*

  .. parsed-literal::

       *gc* args = *mu* value
         value = lattice-gas chemical potential (energy units, may be an equal-style variable)
       *source* args = *rate* value
         value = number of unconditional insertions per invocation

* zero or more keyword/value pairs may be appended
* keyword = *region* or *tfac_insert* or *charge* or *maxspecies* or *overlap_cutoff*

  .. parsed-literal::

       *region* value = region-ID = restrict sites and deletable atoms to the region
       *tfac_insert* value = temperature scale factor for inserted-atom velocities (default 1.0)
       *charge* value = charge assigned to inserted atoms (charge units)
       *maxspecies* value = skip insertions above this many species atoms
       *overlap_cutoff* value = reject insertions at sites with clearance below this
                                without an energy evaluation (default 0.0 = off)

Examples
""""""""

.. code-block:: LAMMPS

   compute SITES all sites/voronoi rmerge 0.3 rmin 1.4 rmax 2.5 occcut 0.9 exclgroup hyd
   fix MC all mc/sites 100 200 2 29494 500.0 sites c_SITES mode gc mu -2.40
   fix IRR all mc/sites 1000 0 He 74839 300.0 sites c_SITES mode source rate 5
   fix VAL all mc/sites 1 50 2 12345 300.0 sites file octsites.txt mode gc mu 0.05

Description
"""""""""""

.. versionadded:: TBD

This fix performs occupancy Monte Carlo of an interstitial species on a
dynamic, geometry-derived site catalogue, in hybrid MD/MC simulations:
grand-canonical insertions/deletions (mode *gc*) or fixed-rate
insertions (mode *source*, irradiation-style) of atoms of the given
type at the sites provided by a
:doc:`compute sites/voronoi <compute_sites_voronoi>` (or a static site
list).  Target applications include hydrogen charging of metals, helium
bubble growth, and general lattice-gas sampling on sites that follow
the evolving microstructure.  In contrast to :doc:`fix gcmc <fix_gcmc>`,
which inserts at random positions in a region, insertions are proposed
only at geometrically favorable interstitial positions, which raises
acceptance rates dramatically in condensed phases.

Every *Nevery* steps the fix builds a *frozen catalogue*: the empty
sites reported by the compute (re-invoked at that step, so the
catalogue follows the current atomic configuration) or read from the
file, plus one *occupied* entry for every current atom of the MC
species in the fix group (and region, if given) -- the species atoms
themselves are the occupied sites; there is no proximity matching.
All trials of the block then use only this frozen catalogue.

In *gc* mode, each of the *Ntrials* trials picks one catalogue entry
uniformly at random.  An empty entry proposes an insertion at its
position; an occupied entry proposes the deletion of its atom.  The
acceptance follows the symmetric lattice-gas convention:

.. math::

   P_\text{ins} = \min\left[1, e^{-\beta (\Delta U - \mu)}\right] \qquad
   P_\text{del} = \min\left[1, e^{-\beta (\Delta U + \mu)}\right]

with :math:`\beta = 1/k_B T` and :math:`\Delta U` the total potential
energy change of the trial, evaluated by a full energy computation
(like the *full_energy* option of :doc:`fix gcmc <fix_gcmc>`; this is
required for EAM, ACE, GRACE, and other many-body potentials).  Note
that :math:`\mu` uses the fix's own lattice-gas reference: the
ideal-gas reservoir factors of :doc:`fix gcmc <fix_gcmc>` (thermal
wavelength, volume, :math:`1/(N+1)`) are intentionally absent, so the
value of :math:`\mu` differs from fix gcmc's chemical potential by a
constant that must be calibrated externally when an absolute reference
is needed.  Isotherm shapes and phase plateaus do not depend on this
constant.  Within a block, a deletion leaves its entry as empty-at-the-
deleted-position (so the reverse move exists) and an accepted insertion
marks its entry occupied; detailed balance is exact within a block, and
the block-to-block catalogue rebuild is a controlled approximation that
vanishes on a static lattice.

In *source* mode, each invocation picks *rate* distinct empty entries
at random and inserts unconditionally (no acceptance test), skipping
(and counting) sites that fail the *overlap_cutoff* guard or when the
catalogue is exhausted.

Inserted atoms are assigned to the fix group (and "all"), given
Maxwell-Boltzmann velocities at *Temp* times *tfac_insert*, and the
*charge* value if the atom style supports charge.  *Temp* and *mu* may
be equal-style variables, re-evaluated at every block, enabling e.g.
simulated annealing of the chemical potential.

Multiple mc/sites fixes may be stacked, one per species (each with its
own type, mu, and site compute); two instances driving the same atom
type are rejected.

.. note::

   The MC species atoms must not be time-integrated during the MC
   block itself (the fix acts at the reneighboring step, before the
   force computation).  Between blocks, any integrator may act on
   them, including the host atoms; the catalogue is rebuilt from the
   current geometry at every block.

.. note::

   The pair styles of the GRACE machine-learning potential family in
   this LAMMPS distribution expose an energy-only evaluation switch;
   this fix detects it automatically (via *pair->extract()*) and
   activates it for the duration of every MC block, which skips the
   force computation in the TensorFlow graph and substantially reduces
   the per-trial cost.  For all other pair styles the full-energy
   evaluation computes (and discards) forces, exactly like
   :doc:`fix gcmc <fix_gcmc>`.

Restart, fix_modify, output, run start/stop, minimize info
""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

This fix writes the state of the random number generators and the move
counters to :doc:`binary restart files <restart>`.  The site catalogue
itself is derived state and is rebuilt at the first block after a
restart.

This fix computes a global vector of length 8, accessible by various
:doc:`output commands <Howto_output>`: (1) trial attempts, (2) accepted
insertions, (3) accepted deletions, (4) current number of species atoms
in the catalogue, (5) catalogue size M (empty + occupied), (6)
instantaneous site concentration = (4)/(5), (7) overall acceptance
ratio, (8) skipped source-mode insertions.  The vector values are
intensive.

No global or per-atom quantities are relevant for minimization; this
fix is not invoked during :doc:`energy minimization <minimize>`.

Restrictions
""""""""""""

This fix is part of the MC package.  It is only enabled if LAMMPS was
built with that package.  See the :doc:`Build package <Build_package>`
page for more info.

Requires a 3d simulation, atom IDs, per-type masses, and reneighboring
(it cannot be used with *neigh_modify once yes*).  An atom map is
created automatically if absent.  The species type must have a mass
even if only inserted atoms carry it.

The static *file* site list assumes the species atoms do not move
between blocks (validation mode): a file site currently holding a
species atom is recognized as occupied by exact position match.  With
moving atoms, use a dynamic compute catalogue and its *occcut* keyword
instead.

Like fix gcmc's *full_energy* mode, the total energy is recomputed for
every trial, which is expensive for large systems and dominates the
cost for machine-learning potentials; choose *Ntrials* and *Nevery*
accordingly.  The MC part is serial (energy evaluation is parallel).
Not supported in v1: Kokkos/GPU/INTEL suffix versions, ReaxFF and
charge-equilibration potentials, core-shell models, molecule insertion,
2d simulations.  When used with kspace, inserted charges change the
total system charge; the usual caveats of charged-system Ewald sums
apply.

Related commands
""""""""""""""""

:doc:`compute sites/voronoi <compute_sites_voronoi>`,
:doc:`fix gcmc <fix_gcmc>`, :doc:`fix atom/swap <fix_atom_swap>`,
:doc:`fix widom <fix_widom>`

Default
"""""""

tfac_insert = 1.0; no charge assignment; no region; no maxspecies cap;
overlap_cutoff = 0.0 (off).
