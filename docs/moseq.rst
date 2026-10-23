keypoint-MoSeq syllables
========================

The second temporal segmentation of the SLEAP data, next to the k-means
prototypes: keypoint-MoSeq (Weinreb et al. 2024) fits an autoregressive
switching model to the keypoints and returns one syllable per frame.  TadPose
only prepares the inputs and drives the fit; keypoint-MoSeq and JAX live in
their own conda environment and are never a package dependency.

.. contents::
   :local:
   :depth: 1


Input: confidence only where observed
-------------------------------------

:mod:`tadpose.moseq_export` reads the stage-1 CSVs
(:doc:`sleap_pipeline`).  A node keeps its raw SLEAP score as confidence only
where its state is ``STATE_OBSERVED``.  Everything stage 1 did not observe has
confidence 0:

* readings a stage-1 gate rejected (low score, eye on the gut, over-long
  segment, gut teleport);
* gaps stage 1 filled by linear interpolation;
* hidden eyes, whether placed by body-frame interpolation or copied onto the
  visible eye.

keypoint-MoSeq adds a small pseudocount and gives such points a very large
noise variance, so the likelihood rests on observed readings only and the AR
model does not learn our fills as dynamics.  The filled *position* is kept as
the starting value.  Left as NaN, keypoint-MoSeq interpolates it linearly in
plate coordinates; across a side-lying stretch of thousands of frames that
put the hidden eye up to 99 px from the gut (median 12 px) and the pose PCA
spent most of its variance on these fake poses.  Nodes stage 1 left empty
(both eyes lost, long gut or tail gaps) stay NaN.

The PCA is fitted on frames in which every node has confidence above 0.5
(``exclude_outliers_for_pca``).

One npz per video holds ``coords`` (wells, T, 7, 2) in plate pixels and
``conf`` (wells, T, 7) on the full 50 fps slot grid, so a syllable index is a
slot index and joins ``rows.parquet`` on (video, well, slot).  A recording is
one animal: ``<video_stem>__<well>``.

All seven nodes are modelled (the tail is the effector); heading runs from the
tail base (posterior) to the eyes (anterior).


Fit set
-------

The full data (51.5 M frames) does not fit on a GPU, so the model is fitted on
one contiguous window per animal: 7,000 frames (140 s) from each of the 288
animals, 2.0 M frames in all.  Window starts are uniform and seeded; a window
whose gut is missing in more than half its frames is redrawn (up to 20 times).
Two draws (seeds 0 and 1) give the stability check.  The windows are listed in
``fitset_seed<n>.csv`` with the missing share per node.


Body size
---------

keypoint-MoSeq centres and rotates every frame but does not rescale it, so a
plate of bigger animals reads as a different pose.  Body length (eye midpoint
to gut to tail tip) varies by about 9 % between animals on a plate and by
about 20 % between plates, while the pixel scale is the same in every video
(7.0 px/mm ± 3 %).  :mod:`tadpose.moseq_scale` therefore scales each animal to
a common body length (``REFERENCE_BODY_LENGTH_PX``) before the fit-set draw.

Size is measured on the animal's reference (drug-free) recordings only, as the
95th percentile of its body length over fully observed frames.  Under drug the
same animals stay 5-15 % shorter even at that percentile (rolling, pitching,
curling); that shortening is posture and must survive the normalisation, so a
per-recording size would erase part of the drug effect.

The animal of each recording comes from ``<moseq>/animals.csv`` (columns
``recording, animal, is_reference``), written by a dataset-specific driver.
Without the table the fit sets keep raw pixel sizes; with it the draw writes
``fit_sets/body_scale.csv`` and records each window's factor.


PCA and the kappa scan
----------------------

:mod:`tadpose.moseq_fit` ``pca`` creates the keypoint-MoSeq project, fits the
pose PCA and sets the latent dimension (scree in ``pca_scree.csv``).  The 90 %
rule gives 3 components; we use 4 (94 %), fixed in ``LATENT_DIM``, so that
the posture detail of slow, erratic spasms is not discarded.

``kappa-scan`` fits the AR-HMM alone for 50 iterations at each kappa from
10\ :sup:`2` to 10\ :sup:`8` and writes the median syllable duration it
produces.  Kappa is the stickiness prior; it is not learned.  The target
durations are 3, 6, 12 and 25 frames (60 to 500 ms); which one to use is
decided against C-bend counts, treatment separation and stability, not by
default.  The full model then needs a lower kappa than the AR-HMM.

The full model
--------------

``calibrate`` continues an AR-only start (the scan's checkpoint at that kappa,
or a new 50-iteration fit) as the full keypoint-SLDS for 200 iterations, once
at the AR kappa and once at each of four lower decades
(``FULL_KAPPA_FACTORS``); fits that already have results are skipped.  kpms
advises a lower kappa for the full model, and on the tadpoles its syllables
ran 3-6 times longer than the AR model's at the same kappa.  The
target is the mean bout of the pdv k-means PMs after the 2-frame merge, so that
syllables and PMs describe the same timescale.  ``full`` then fits the chosen
kappa for 500 iterations.  Each fit writes its syllable statistics to
``full_fits/<model>.json``.

Not yet done: the noise calibration (it needs about 50 hand-annotated
frames), so the confidence-to-noise mapping is keypoint-MoSeq's default.


Running it
----------

See ``slurm/README.md``: ``bash slurm/submit_moseq.sh`` chains the export
array, the fit-set draw, the PCA and the kappa-scan array.  Outputs go to
``<sleap_output_root>/moseq/``.
