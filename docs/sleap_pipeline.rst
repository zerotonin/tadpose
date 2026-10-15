SLEAP pipeline: from tracking CSV to clustering matrix
======================================================

This page documents how Caroline Beck's 2026 SLEAP re-tracking becomes the
TadPose clustering matrix, one stage at a time, and how any cluster label can be
traced back to the frame of the original video it came from.

.. contents::
   :local:
   :depth: 2


Overview
--------

.. code-block:: text

   stage 0  raw SLEAP export          <video>_chunkNNN_fA-B_wells.csv   (Caroline's export_csv.py)
      │       sleap_ingest            join chunks, uniform time grid, one track per well
      ▼
   stage 1  interpolated CSV          <video>_interpolated.csv          sleap_cleaning
      │       gate bad nodes, copy the hidden eye, fill short gut/tail gaps
      ▼
   stage 2  feature CSV               <video>_features.csv              sleap_features
      │       frons, thrust/slip/yaw, aligned posture, per-frame diffs; NaN + omit_reason
      ▼
   stage 3  clustering bundle         clustering/                       sleap_matrix
              usable rows only, row map, omitted-row log, mu/sigma, z-scored matrix

Stages 0 to 2 are per video; stage 3 pools all videos of a clustering cohort.
Every stage is written to disk, so each can be re-run from the previous one and
inspected on its own.


Skeleton and body geometry
--------------------------

Node names are kept exactly as in the SLEAP project:
``left_eye, right_eye, centre, tail1, tail2, tail3, tail_tip``.

* ``centre`` is the **gut**, not the tail base; ``tail1`` is the tail base.
* **Frons** = midpoint of the two eyes. If only one eye is visible, that eye is
  the frons (a side-swimming tadpole hides one eye directly below the other).
* **Posture axis** = gut → frons (``posture_angle_rad``); the aligned posture puts
  the gut on +x, so ``centre_y`` is 0.
* **Kinematics are DLC-style**, so the SLEAP and DLC runs measure the same thing:
  heading (``heading_rad``) = tail base (``tail1``) → frons; thrust and slip
  describe the mean of both eyes and the tail base (``speed_point_*``). The
  three-point mean damps single-node jitter, and the long heading axis keeps
  pixel noise out of yaw. A first version measured at the gut with a gut → frons
  heading; it gave 20 to 40 % wider shoulders in all three speed distributions
  than the (fps-corrected) DLC run, i.e. more small-amplitude jitter.


Well scale (``px_per_mm``)
--------------------------

SLEAP re-tracked the same videos the ring CNN already measured (DB videos
20 to 31), from a plate crop that was not rescaled: the frons position range of
the same well agrees within 3 % between the DLC and SLEAP tracks. The ring-CNN
radius of every well therefore carries over unchanged, and
:mod:`tadpose.sleap_geometry` turns the existing ``geometry_exp*.json`` into one
table (``video_name, well, r_px, px_per_mm = 2 r / 15.6``; 6.5 to 7.5 px/mm on
the Sept-2024 plates). The stored centres are in the old per-well crop frame,
not plate coordinates; the features need only the scale.


Stage 0: raw SLEAP export
-------------------------

One CSV per five-minute chunk, one row per tadpole per video frame. Columns:
``frame, time_s, frame_interval_ms, after_gap, well, instance_score`` and
``<node>_x, <node>_y, <node>_score`` for the seven nodes, in pixels of the
cropped plate video (866 × 574). Fixed plate marks and duplicates were removed
upstream (``assign_wells.py``). Two things are implicit in this format:

* a **missed detection** is a missing row, not a row of NaN;
* a **camera-dropped frame** has no row at all; it only shows as a longer
  ``frame_interval_ms`` / ``after_gap = 1``.

The files are never modified.


Stage 1: the interpolated CSV
-----------------------------

:mod:`tadpose.sleap_ingest` joins the chunks (checked to be contiguous) and builds
a **uniform 50 fps grid of slots** per video. Each video frame gets one slot; a
camera drop of *n* frames gets *n* placeholder slots (``is_dropped = True``,
``frame`` empty). Every well gets every slot, so slot *k* is the same instant in
all 24 wells. :mod:`tadpose.sleap_cleaning` then applies, in this order:

1. **Gates** reject a node reading that is evidently wrong (never because of
   where the animal is in the well):

   ========================  ==========================================================
   gate                      rule (constants in :mod:`tadpose.sleap_constants`)
   ========================  ==========================================================
   low score                 node score < ``MIN_NODE_SCORE`` (0.3)
   eye on the gut            eye within ``EYE_ON_CENTRE_PX`` (3 px) of ``centre``
   over-long segment         segment > ``MAX_SEGMENT_FACTOR`` (2) × the animal's median
                             length rejects its distal node (tail tip on the well rim)
   teleport                  gut jumps > ``MAX_CENTRE_JUMP_PX`` (30 px) in and straight
                             back out: the whole frame is rejected
   ========================  ==========================================================

2. **Hidden eye**: if exactly one eye is valid, the other takes its x/y.
3. **Gap filling**: gaps in ``centre`` and the tail nodes are filled linearly if
   they are at most ``MAX_INTERP_FRAMES`` (25 slots = 0.5 s) long, bounded by
   valid values on both sides, and contain no camera-dropped slot.

**Never filled:** frames with **both eyes missing** (no frons, so no posture),
and **camera-dropped slots**. They stay NaN.

Columns of ``<video>_interpolated.csv`` (one row per well and slot):

* ``video_stem, well, well_number, slot, frame, raw_frame, time_s, is_dropped,
  is_detected, instance_score``
* per node: ``<node>_x, <node>_y`` (cleaned, px), ``<node>_score`` (raw SLEAP
  score), ``<node>_state``, ``<node>_gated``

``<node>_state`` codes: ``0`` observed · ``1`` interpolated · ``2`` eye copied
from the other eye · ``3`` missing. ``<node>_gated`` is ``True`` where a gate
rejected the raw reading (its value may since have been interpolated).
``raw_frame = frame + skip_start_frames`` is the frame index in the cropped video
(the older recordings dropped 10 frames before tracking).


Stage 2: the feature CSV
------------------------

:mod:`tadpose.sleap_features` computes, for every slot:

* ``speed_point_x_px, speed_point_y_px, frons_x_px, frons_y_px, heading_rad,
  posture_angle_rad``;
* **velocity**: ``thrust_mm_s`` (forward) and ``slip_mm_s`` (leftward) of the
  speed-point displacement since the previous slot, rotated into the current
  heading frame;
  ``yaw_rad_s`` = wrapped heading change, so turning through ±π gives no 2π spike.
  Divided by the real slot interval (``time_s``) and the well's ``px_per_mm``;
* **posture** ``<coord>_mm``: frons at the origin, gut on the +x axis, in mm.
  The frons coordinates and ``centre_y`` are constant (0) and dropped, which leaves
  13: ``left_eye_x/y, right_eye_x/y, centre_x, tail1_x/y, tail2_x/y, tail3_x/y,
  tail_tip_x/y``;
* **posture dynamics** ``<coord>_diff``: per-frame change of each posture
  coordinate (mm per frame), so every diff row lines up with the speed row of
  the same frame.

A row that cannot carry features keeps NaN features and an ``omit_reason``
(first that applies):

======================  ==================================================  ==========================
``omit_reason``         meaning                                             posture / velocity + diffs
======================  ==================================================  ==========================
``camera_dropped``      placeholder slot, no video frame                    NaN / NaN
``no_eyes``             both eyes missing: no frons                         NaN / NaN
``missing_node``        a gut/tail node gap too long to fill                NaN / NaN
``no_previous``         previous slot unusable, so no diff or speed         valid / NaN
(empty)                 usable                                              valid / valid
======================  ==================================================  ==========================

``is_usable`` is ``True`` exactly when ``omit_reason`` is empty.


Stage 3: the clustering bundle
------------------------------

:mod:`tadpose.sleap_matrix` pools the stage-2 tables of a cohort and keeps only
usable rows. **Rows are left out before z-scoring**, so mu and sigma describe
real data only. The 16 clustering features, in order:
``thrust_mm_s, slip_mm_s, yaw_rad_s`` + the 13 ``<coord>_diff`` columns.

========================  ==============================================================
file                      content
========================  ==============================================================
``features_raw.npy``      (N × 16) float32, physical units
``features_zscored.npy``  (N × 16) float32, ``(x − mu) / sigma``: the k-means input
``features_vel_zscored``  (N × 3) the velocity-only feature set (first three columns)
``rows.parquet``          row *i* of both matrices: ``video_stem, well, well_number,
                          slot, frame, raw_frame, time_s``
``omitted_rows.parquet``  every stage-2 row left out, same keys + ``omit_reason``
``musigma.csv``           per feature: ``feature_index, feature, mu, sigma``
                          (population sigma); reused unchanged on new data (genes)
``omission_summary.csv``  per video and well: kept rows, omitted rows per reason
``manifest.json``         counts, feature order, videos, cleaning parameters
========================  ==============================================================


Running it
----------

Locally, with explicit paths (or the ``local`` profile keys
``sleap_predictions_root``, ``sleap_geometry_csv``, ``sleap_output_root``):

.. code-block:: bash

   tadpose sleap-features --list --predictions <predictions folder>
   tadpose sleap-features --all  --predictions <predictions folder> \
       --geometry <sleap_well_geometry.csv> --out <output root>
   tadpose sleap-matrix --out-root <output root> --expect 12

The start-frame trim of each video is read from Caroline's ``batch_log.txt``
(``skip_frames=``); without that log ``--skip-start-frames`` must be given.
On the cluster, ``bash slurm/submit_sleap.sh`` runs one array task per video
and chains the merge after all of them succeed (see ``slurm/README.md``).
Output layout under the output root:

.. code-block:: text

   stage1_interpolated/<video>_interpolated.csv
   stage2_features/<video>_features.csv
   clustering/        features_raw.npy, features_zscored.npy,
                      features_vel_zscored.npy, rows.parquet,
                      omitted_rows.parquet, musigma.csv, omission_summary.csv,
                      manifest.json (includes code version + git commit)
   logs/              SLURM task and merge logs


Tracing a cluster label back to the footage
-------------------------------------------

Cluster labels are produced in the row order of ``features_zscored.npy``, so
label *i* belongs to ``rows.parquet`` row *i*:

.. code-block:: python

   import numpy as np
   import pandas as pd

   rows = pd.read_parquet("clustering/rows.parquet")
   labels = np.load("labels_k28.npy")
   hits = rows[labels == 7]            # every frame assigned to prototype 7
   hit = hits.iloc[0]
   # hit.video_stem, hit.well and hit.raw_frame locate the frame in the
   # cropped plate video; hit.slot finds the row in both stage CSVs.

The stage-1 row (same ``video_stem, well, slot``) shows whether any node of that
frame was gated, interpolated or eye-copied; the stage-0 chunk with the same
``frame`` shows the untouched SLEAP output.


Measured on the Sept-2024 plates
--------------------------------

Two videos, provisional scale 7.5 px/mm (run before the ring geometry was wired in;
the real per-well scale is ~7.0, so speeds there read ~7 % high):

==================================  ===================  ==================
                                    4-AP, frog D         baseline (no drug)
==================================  ===================  ==================
eye copied (per eye)                ~21 %                0.1 %
eyes gated (score / position)       ~2.4 %               0.05 %
tail tip interpolated               1.5 %                0.23 %
rows usable                         99.25 %              99.95 %
omitted: no_eyes                    0.55 %               0.02 %
omitted: missing_node               0.08 %               0.00 %
omitted: camera_dropped             0.03 %               0.00 %
==================================  ===================  ==================

Per video, stage 1 + 2 take about 3 minutes; the CSVs are about 1.2 GB
(stage 1) and 1.7 GB (stage 2).
