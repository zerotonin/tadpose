Overview
========

What problem does TadPose address?
----------------------------------

Developmental and epileptic encephalopathies (DEE) are severe, early-onset
seizure disorders caused by *de novo* mutations in over 140 genes.  Each
genotype is associated with a distinct constellation of developmental
disruptions, and anti-seizure medications frequently fail to control
seizures or target the underlying defect.  Distinguishing one genetic or
pharmacological phenotype from another therefore requires behavioural
assays with both high temporal resolution and the throughput to compare
many animals across many conditions.

Conventional behavioural readouts collapse continuous motor output into a
handful of summary statistics — mean velocity, turn frequency, distance
travelled.  These measures capture gross changes in activity but discard
the temporal and postural detail in which brief or low-probability
seizure actions carry mechanistic meaning.  *Xenopus laevis* tadpoles are
well suited to closing this gap: the model permits targeted genome
editing, transdermal drug delivery, and high-throughput imaging in
multi-well formats.

TadPose extends the unsupervised behavioural-clustering framework of
Braun *et al.* (2010), which segments continuous movement into data-driven
"behavioural prototypes" without human-defined categories.  Earlier
applications of that framework treat the animal as a single moving point;
TadPose adds multi-landmark **posture dynamics** alongside centre-of-mass
kinematics, recovering subtle motor patterns that point-tracking alone
cannot resolve.

The pipeline
------------

The analysis runs as a sequence of independently re-runnable stages, with
file or database intermediates between them:

1. **Well detection** (:mod:`tadpose.well_detection`) — a Hough circle
   transform with eigenvector-corrected centres localises all 24 wells
   despite lens distortion.
2. **Video segmentation** (:mod:`tadpose.video_segmentation`) — full-plate
   recordings are split into per-well clips for pose estimation.
3. **Pose estimation** — seven anatomical landmarks (left eye, right eye,
   tail base, three tail segments, tail tip) are tracked with
   `DeepLabCut <https://github.com/DeepLabCut/DeepLabCut>`_
   (:mod:`tadpose.dlc_runner`).
4. **Feature extraction** (:mod:`tadpose.feature_extraction`) — centre-of-
   mass displacement is decomposed into **thrust** (forward), **slip**
   (lateral) and **yaw** (rotational) components, and posture is aligned to
   a frons-to-tail-base body axis so that frame-wise landmark
   displacements describe shape change independently of position and
   heading.
5. **Cleaning and normalisation** (:mod:`tadpose.feature_cleaning`,
   :mod:`tadpose.normalisation`) — distribution-based thresholds remove
   tracking artefacts and all features are z-scored.
6. **Behavioural clustering** (:mod:`tadpose.clustering`) — GPU-accelerated
   k-means partitions the combined velocity and posture-dynamics features
   into behavioural prototypes.  The number of clusters is chosen from a
   quality (Calinski–Harabasz, Davies–Bouldin) versus stability trade-off
   evaluated across data-deletion replicates.
7. **Post-clustering analysis** (:mod:`tadpose.analysis`) — per-trial
   cluster proportions, group comparisons via
   `reRandomStats <https://github.com/zerotonin/reRandomStats>`_, hidden
   Markov-chain transition structure, and centroid visualisation.

What the method recovers
------------------------

Clustering the combined posture-and-velocity features resolves **36 stable
behavioural prototypes**, which fall into seven qualitatively distinct
categories: C-shaped contractions (and a plate-edge variant), uncoordinated
tail bends, impact compressions, head bobbing, body flips, and regular
versus undulatory swimming, alongside a stationary state.  Velocity-only
clustering of the same data yields just eight prototypes that capture
general swimming but miss the seizure-specific motor disruptions — a
roughly fivefold gain in discriminative resolution from adding posture.

Because each prototype carries enough specificity, individual tadpoles can
be classified by treatment through reverse inference on their prototype
abundances.  The framework has been applied to three seizure models: a
pentylenetetrazol (PTZ) dose–response series, a 4-aminopyridine (4-AP)
challenge with valproate (VPA) rescue, and a CRISPR F0 haploinsufficiency
model of the DEE-linked gene *NeuroD2*.

Provenance
----------

The method and its proof-of-concept application were developed by
Alexander R. H. Matthews in a BSc(Hons) thesis (University of Otago,
Department of Zoology), supervised by Bart R. H. Geurten, with the dataset
and genetic models provided by Caroline Beck.  This package is the
publication-ready refactor of that research codebase; a companion
manuscript is in preparation.
