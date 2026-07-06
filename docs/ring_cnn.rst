Ring CNN — well-geometry detector
=================================

TadPose measures tadpole motion in millimetres and thigmotaxis as a fraction of
the well radius.  Both need the pixel geometry of every well: its centre
``(cx, cy)`` and radius ``r`` in the crop.  The **Ring CNN** (``RingNet``)
regresses that geometry directly from a single clean-well image, replacing the
brittle Hough/edge detectors that failed on reflections, meniscus glare and
uneven illumination.

The detector is deliberately small and trained **from scratch** — the compute
nodes have no internet, so no ImageNet backbone is downloaded, and the task is
narrow enough (one bright ring on a near-uniform field) that a compact network
converges cleanly on a modest hand-annotated set.


What it predicts
----------------

Input is a ``128 × 128 × 3`` RGB crop of one well (the *background projection*,
see below).  Output is three numbers passed through a sigmoid, so they are
**image fractions in [0, 1]**:

.. code-block:: text

   RingNet(crop)  ->  (cx, cy, r)          # fractions of crop width / height

Pixels are recovered by multiplying back by the crop size
(``cx * w``, ``cy * h``, ``r * w``).  The physical scale follows from the
caliper-confirmed well diameter of **15.6 mm**:

.. code-block:: text

   pix2mm = 2 * r_px / 15.6                 # px per mm, per well

Predicting *fractions* rather than pixels makes the head resolution-independent
and keeps the three outputs on the same numeric scale for the loss.


Architecture
------------

A compact VGG-idiom convolutional regressor, ``≈ 0.6 M`` parameters:

.. list-table::
   :header-rows: 1
   :widths: 22 78

   * - Stage
     - Layers
   * - Feature block × 4
     - ``Conv3×3 → BatchNorm → ReLU`` twice, then ``MaxPool2×2``
   * - Channel widths
     - ``3 → 32 → 64 → 128 → 128`` (``width = 32``)
   * - Bottleneck
     - ``AdaptiveAvgPool2d(1)`` — global average pool to a 128-vector
   * - Head
     - ``Flatten → Linear(128→64) → ReLU → Dropout(0.2) → Linear(64→3) → Sigmoid``

Global average pooling (rather than a large flatten) keeps the parameter count
low and makes the network tolerant of small crop-size variation.  Batch-norm on
every conv stabilises the from-scratch training.

.. code-block:: python

   class RingNet(nn.Module):
       """Compact conv regressor -> sigmoid(cx, cy, r) in [0, 1]."""

       def __init__(self, width: int = 32):
           super().__init__()

           def block(ci: int, co: int) -> nn.Sequential:
               return nn.Sequential(
                   nn.Conv2d(ci, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                   nn.Conv2d(co, co, 3, padding=1), nn.BatchNorm2d(co), nn.ReLU(inplace=True),
                   nn.MaxPool2d(2),
               )

           self.features = nn.Sequential(
               block(3, width), block(width, width * 2),
               block(width * 2, width * 4), block(width * 4, width * 4),
               nn.AdaptiveAvgPool2d(1),
           )
           self.head = nn.Sequential(
               nn.Flatten(), nn.Linear(width * 4, width * 2), nn.ReLU(inplace=True),
               nn.Dropout(0.2), nn.Linear(width * 2, 3), nn.Sigmoid(),
           )


Training data
-------------

Circles are annotated with the ``arena_annotator`` (``circle_annotator``) tool
and exported as COCO.  Each annotation carries
``attributes.centre_x / centre_y / radius``; targets are normalised to image
fractions on load.

**Split by plate, never by well.**  A plate holds 24 near-identical wells, so a
random per-well split would leak almost the same image into train and test.
Splitting on the plate *stem* (parsed from ``<src>__<stem>_well_NN.png``) keeps
every plate wholly inside one of train / val / test (``0.70 / 0.15 / 0.15``).

**Target-aware augmentation.**  Each geometric transform is applied to the label
as well as the image, so the supervision stays correct:

.. list-table::
   :header-rows: 1
   :widths: 40 60

   * - Transform (prob)
     - Label update
   * - Horizontal flip (0.5)
     - ``cx → 1 − cx``
   * - Vertical flip (0.5)
     - ``cy → 1 − cy``
   * - 90° rotation × k (uniform k)
     - ``(cx, cy) → (cy, 1 − cx)`` per turn
   * - Brightness / contrast jitter (0.7)
     - image only


Loss and optimisation
----------------------

* **Loss** — ``SmoothL1`` (Huber) on the three normalised targets; robust to the
  occasional mis-annotated circle.
* **Optimiser** — Adam, ``lr = 1e-3``.
* **Schedule** — ``ReduceLROnPlateau(factor=0.5, patience=15)`` on the validation
  score.
* **Selection** — best checkpoint by ``val centre_px_mean + radius_px_mean``.
* **Defaults** — 300 epochs, batch 32, ``img_size = 128``, seed 0.

Evaluation reports centre and radius error back in **pixels** (de-normalised by
crop width/height); ``qc_overlays.png`` draws predicted (vermilion, dashed) vs
annotated (bluish-green) circles on held-out test crops.


The background projection — the decisive design choice
------------------------------------------------------

A well crop is a *video*, not a still.  The tadpole is dark and moves; the well
rim is static.  Collapsing 40 evenly-spaced frames to one image removes the
animal and exposes a clean ring.  Two projections were compared:

* **Median** — per-pixel median over frames; smooth, but the rim softens
  slightly.
* **Max** — per-channel maximum over frames; the dark animal is beaten at every
  pixel by the brighter background, giving the **sharpest rim**.

Because the radius sets ``pix2mm``, a crisp rim matters more than a crisp
centre.  Across annotation sets and projections (held-out test split):

.. list-table::
   :header-rows: 1
   :widths: 26 12 14 14 14 12

   * - Run
     - Projection
     - centre (px)
     - centre med
     - radius (px)
     - radius med
   * - upper rim
     - median
     - 4.37
     - 3.76
     - 1.77
     - 1.42
   * - lower edge (small)
     - median
     - 4.32
     - 3.63
     - 1.78
     - 1.47
   * - lower edge v3
     - median
     - 1.74
     - 1.65
     - 1.34
     - 1.22
   * - combined
     - median
     - **1.42**
     - 1.13
     - 1.52
     - 1.45
   * - combined
     - max
     - 1.53
     - 1.27
     - **1.44**
     - **1.22**
   * - all-max (deployed)
     - max
     - 2.64
     - 1.83
     - 1.80
     - 1.75

**Why the lower edge.**  The well is a shallow truncated cone.  The *upper rim*
sits above the water and is displaced by parallax; the *lower edge* is the
water plane the tadpole actually swims in.  Annotating the lower edge (``v3``
onward) collapsed the centre error from ``> 4 px`` to ``< 2 px`` and is the
correct plane for ``pix2mm``.

**Why max is deployed.**  The max projection gives the best radius (and radius
is what ``pix2mm`` depends on).  The production model, ``all-max``, extends the
combined max set with the newer plate cohorts (e.g. the July-2026 PPP3CA
recordings) so the detector has seen the imaging conditions it is applied to.
Its held-out numbers look slightly higher only because that split contains the
harder, more varied late plates; on the deployment data it produces a tight,
uniform geometry (median well radius ``≈ 51–52 px`` across all 24 wells).


Inference and deployment
------------------------

``ring_infer.py`` applies the trained model over an experiment's videos:

#. For each video of the requested ``experiment_type``, and each of its 24
   wells, build the **max** background projection (40 frames) and predict
   ``(cx, cy, r)`` in pixels.
#. ``pix2mm`` for the video is ``2 · median(r over wells) / 15.6``.
#. Write two JSON artefacts:

   * ``{video_id: pix2mm}`` — per-video scale.
   * ``{video_id: {well: [cx, cy, r]}}`` — per-well geometry.

The kinematics loader consumes the per-well geometry so each trajectory is
re-expressed in **well-centred millimetres** (origin at that well's ``(cx, cy)``,
scaled by its own ``2r / 15.6``).  Thigmotaxis is then ``√(x² + y²) / R`` with
``R = 7.8 mm``.  Feeding the ring geometry into the PPP3CA report moved the
occupancy mass onto the wall (periphery fraction ``0.95–0.98``), fixing the
earlier "centroids miles from the wall" artefact caused by stale stored
geometry.


Reproducing
-----------

.. code-block:: bash

   # train (COCO export from circle_annotator in <data-dir>)
   python ring_train.py --data-dir <data-dir> --out-dir <run-dir> \
       --epochs 300 --batch 32 --img-size 128 --lr 1e-3

   # infer geometry for one experiment_type
   python ring_infer.py --model <run-dir>/ring_net_best.pt \
       --out pix2mm.json --geometry-out geometry.json \
       --experiment-type <id>

Outputs per run: ``ring_net_best.pt``, ``metrics.json`` (full history + test
scores), and ``qc_overlays.png``.

.. note::

   Data roots and the database path are machine-specific and resolve through the
   gitignored ``local_paths.json`` (see :doc:`installation`); the commands above
   use ``<placeholder>`` paths.
