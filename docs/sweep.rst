k-means stability sweep and k-selection
=======================================

The behavioural prototypes are k-means centroids.  Which *k* to use is
decided from a stability sweep: the same clustering is fitted many times over
a grid of *k*, with parts of the data left out, and five criteria are compared.
:mod:`tadpose.sweep` runs the sweep on a pool of GPUs; :mod:`tadpose.kselect`
turns the finished sweep into the selection evidence.

.. contents::
   :local:
   :depth: 1


The grid
--------

Defined once in :mod:`tadpose.sweep_constants` and identical for every run, so
runs on different tracking data stay comparable:

* **k**: 2 to 40, then 45, 50, ..., 100 (51 values);
* **leave-out size** (``delSize``): 0, 10, 25, 50 % of the rows;
* **leave-out position** (``delPos``): 2, 4, ..., 100 % (50 positions);
* per feature set: 51 × 4 × 50 = 10,200 fits (20,400 for posture+velocity and
  velocity-only together).

Each fit is a full cuML k-means (``k-means||`` initialisation, Lloyd,
euclidean) on all remaining rows (:mod:`tadpose.clustering`).


Leave-out: what the 50 positions do
-----------------------------------

The leave-out (after Alex Matthews' implementation) removes **one contiguous
block** of rows: ``delSize`` % of the matrix, starting ``delPos`` % of the way
through and wrapping round at the end.  At ``delSize`` 0 nothing is removed, so
the 50 positions are simply 50 independent random initialisations of the full
data: they measure how reproducible the clustering is.  At 10, 25 and 50 % the
fits also see different data.

Because the matrix rows are grouped (video, then well, then time), a block is
never a random scatter of frames: it removes whole stretches of animals, and at
50 % whole plates.  This is deliberate.  Dropping random single frames from
tens of millions would leave every centroid unchanged and the test would have
no teeth; dropping animals asks whether a prototype survives when the animals
that show it are partly missing.  Across the 50 positions every part of the
matrix is left out somewhere.


Seeds
-----

Every fit gets a distinct, reproducible k-means seed:
``CRC32("<tag>|<k>|<delSize>|<delPos>")`` (31 bit), checked to be collision-free
over the whole grid.  Python's built-in ``hash()`` of a string is salted per
process and must not be used for this.  The seed, the row count and whether
labels were kept are stored in each fit's meta JSON.


What each fit stores
--------------------

``<sweep>/<tag>/delSize_<ds>/k_<k>/``: the meta JSON (centroids, inertia,
Calinski-Harabasz, seed, timing) and the centroids ``.npy``.  **Labels** are
kept only for one clone per *k* (``delSize`` 0, ``delPos`` 2): the k-selection
needs one full-data labelling per *k*, and leave-out labels do not align with
the row map.  Keeping every fit's labels cost about 1.2 TB in the DLC run.


Running a sweep
---------------

.. code-block:: bash

   # once: describe the sweep (writes <sweep>/sweep_run.json)
   tadpose sweep init --sweep-dir <sweep> --tag-prefix sleap2026 \
       --feature-set pdv=<bundle>/features_zscored.npy \
       --feature-set vel=<bundle>/features_vel_zscored.npy

   # GPU workers (login node; add more at any time, they share one pool)
   bash slurm/submit_sweep.sh workers <sweep> 12

   tadpose sweep status --sweep-dir <sweep>

   # one-screen monitor, refreshed every minute inside a single ssh session
   ssh -t <cluster> 'watch -n 60 <checkout>/slurm/sweep_watch.sh <sweep>'

Workers pull fits from a shared queue on the filesystem: a fit is **done** when
its meta JSON exists, **claimed** by an atomic lock file that is refreshed every
minute (a claim older than five minutes belongs to a dead worker and is taken
over), and **parked** after three failures.  Each worker packs several fits on
its GPU (5 on an H100 or 80 GB A100, 2 on a 24 GB card); the fit is
memory-bandwidth-bound, so throughput scales with more GPUs, not deeper packing.
``--test`` runs a 30-fit sub-grid to check a new setup.


k-selection
-----------

Once ``tadpose sweep status`` reports ``remaining=0``:

.. code-block:: bash

   bash slurm/submit_sweep.sh kselect <sweep>
   # or one feature set locally:
   tadpose kselect --sweep-dir <sweep> --feature-set pdv

Per feature set this writes to ``<sweep>/kselect_<feature set>/``:

* ``kselect_metaData.csv``: every fit with its **instability**, the mean
  Hungarian-matched centroid shift to the most stable clone of the same
  (*k*, ``delSize``) (Geurten et al. 2006; Braun et al.);
* ``selection_summary.csv``: per *k*, Calinski-Harabasz, inertia, the
  **Kneedle elbow**, and **silhouette** and **Davies-Bouldin** on stratified
  subsamples of the labelled clone (silhouette is O(n²), so it is estimated
  on 400 rows per cluster, 8 repeats, giving a median and IQR);
* ``kselect_agreement`` (SVG + PNG + CSV): all five criteria against *k* with
  their dispersion, and the best *k* per criterion side by side.

The choice of *k* is made by reading this figure; the code does not pick it.
