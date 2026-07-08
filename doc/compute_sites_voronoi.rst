.. index:: compute sites/voronoi

compute sites/voronoi command
=============================

Syntax
""""""

.. code-block:: LAMMPS

   compute ID group-ID sites/voronoi rmerge dist keyword values ...

* ID, group-ID are documented in :doc:`compute <compute>` command
* sites/voronoi = style name of this compute command
* rmerge dist = merge Voronoi vertices closer than *dist* (distance units, required)
* zero or more keyword/value pairs may be appended
* keyword = *rmin* or *rmax* or *vmin* or *vmax* or *metric* or *region* or *coord* or *hostgroup* or *occcut* or *exclgroup*

  .. parsed-literal::

       *rmin* value = minimum site clearance (distance units, may be an equal-style variable)
       *rmax* value = maximum site clearance (distance units, may be an equal-style variable)
       *vmin* value = minimum probe Voronoi volume (volume units, may be an equal-style variable)
       *vmax* value = maximum probe Voronoi volume (volume units, may be an equal-style variable)
       *metric* value = *radius* or *volume* = also compute the probe Voronoi volume column
       *region* value = region-ID = only keep sites inside the region
       *coord* values = k rcut = only keep sites with at least k *hostgroup* atoms within rcut
       *hostgroup* value = group-ID = atoms counted by the *coord* criterion (default: the compute group)
       *occcut* value = dist = drop sites within *dist* of any *exclgroup* atom
       *exclgroup* value = group-ID = atoms carrying the *occcut* veto

Examples
""""""""

.. code-block:: LAMMPS

   compute SITES all sites/voronoi rmerge 0.3
   compute OCT all sites/voronoi rmerge 0.3 rmin 1.6 rmax 2.0
   compute BULK all sites/voronoi rmerge 0.3 rmin 1.4 coord 6 3.5 hostgroup metal occcut 0.9 exclgroup hydrogen

Description
"""""""""""

.. versionadded:: TBD

Define a computation that builds a catalogue of *interstitial sites*:
the vertices of the Voronoi tessellation of the atoms in the compute
group.  Each Voronoi vertex is the center of a locally largest empty
sphere; its distance to the nearest atoms (its *clearance*) is by
construction the insertion clearance available at that point.  Because
the sites are derived from the current geometry, the catalogue adapts
to any structure: perfect crystals, defects, voids, gas bubbles, and
surfaces, with no assumed sublattice.

The per-processor tessellation uses the Voro++ library exactly like
:doc:`compute voronoi/atom <compute_voronoi_atom>` (each processor
tessellates its subdomain extended by the ghost communication cutoff).
Vertices are then merged by single-linkage clustering with cutoff
*rmerge*: clusters are replaced by their centroid and the cluster
clearance is the minimum over its members.  Merging is required because
Voronoi vertices of high-symmetry lattices are degenerate and split
into near-coincident vertices by floating-point noise or thermal
displacements.  *rmerge* should be much smaller than the distance
between distinct sites (0.3 Angstroms works well for metals).

The merged sites are then filtered.  A site is kept only if it passes
all active criteria:

* its clearance lies in [*rmin*, *rmax*]
* its probe Voronoi volume lies in [*vmin*, *vmax*] (only if the
  volume metric is active; the probe volume is the volume of the
  Voronoi cell a test point at the site position would have among the
  existing atoms, computed by a local re-tessellation; this matches
  the volume-based site classification of :ref:`(vonPezold) <vonPezold1>`)
* it lies inside *region* (evaluated at build time only)
* at least *k* atoms of the *hostgroup* lie within *rcut* of it
  (*coord* criterion).  This distinguishes internal voids and
  subsurface sites from outer vacuum: vertices generated in vacuum
  regions of non-periodic boxes have large clearance but low
  coordination.
* no atom of *exclgroup* lies within *occcut* of it.  This is the
  intended way to suppress candidate sites that are already occupied
  by an existing (e.g. Monte Carlo) species when the tessellation
  alone does not remove them.

The *rmin*, *rmax*, *vmin*, and *vmax* thresholds may be specified as
equal-style variables (e.g. ``v_myrmin``), re-evaluated at every
invocation, so the filter window may be time-dependent.

The compute is intended as the dynamic site source for
:doc:`fix mc/sites <fix_mc_sites>`, which re-invokes it at every MC
block, but it can be used standalone (e.g. dumped with
:doc:`dump local <dump>`) to analyze interstitial-site distributions.

.. note::

   Each processor only keeps sites whose merged position falls inside
   its own subdomain, so the union over processors is a duplicate-free
   global catalogue.  The tessellation around a site is only complete
   if its clearance is smaller than the ghost-atom communication
   cutoff; a warning is printed when an accepted site violates this.
   Increase the communication cutoff via :doc:`comm_modify <comm_modify>`
   or tighten *rmax* in that case.  In non-periodic directions the
   tessellation is clamped at the box boundary (like
   :doc:`compute voronoi/atom <compute_voronoi_atom>`); use the *coord*
   criterion to exclude the resulting vacuum vertices.

Output info
"""""""""""

This compute calculates a local array with one row per accepted site
owned by this processor.  The number of columns is 5, or 6 if the
volume metric is active: x, y, z, clearance, [volume,] coordination
(the *coord* atom count, 0 if *coord* is not used).  It also calculates
a global scalar, the total number of accepted sites across all
processors.  The scalar is intensive.

The local array can be accessed by any command that uses local values
from a compute as input, e.g. :doc:`dump local <dump>` or
:doc:`fix ave/histo <fix_ave_histo>`.  Positions are in distance
:doc:`units <units>`, the clearance in distance units, the volume in
volume units.

Restrictions
""""""""""""

This compute is part of the VORONOI package.  It is only enabled if
LAMMPS was built with that package.  See the
:doc:`Build package <Build_package>` page for more info.  It requires
a 3d simulation.

Related commands
""""""""""""""""

:doc:`fix mc/sites <fix_mc_sites>`,
:doc:`compute voronoi/atom <compute_voronoi_atom>`

Default
"""""""

No clearance/volume window, no region, no coordination criterion, no
occupancy veto; *hostgroup* = the compute group; only the clearance
metric is computed.

----------

.. _vonPezold1:

**(vonPezold)** von Pezold, Lymperakis, Neugebauer, Acta Materialia 59,
2969 (2011).
