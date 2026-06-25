Usage
=====

Command-line interface
----------------------

Installing the package registers a single ``tadpose`` command that
dispatches to one subcommand per pipeline stage:

.. code-block:: bash

   tadpose --help                 # list the available stages
   tadpose config                 # show the resolved active profile/data root
   tadpose config --export hpc    # emit TADPOSE_* shell vars for the hpc profile
   tadpose <stage> --help         # options for an individual stage

Available stages include ``config``, ``assign-clusters``, ``label``,
``hmm``, ``hmm-groups`` and ``cluster-meta``.  Each stage derives its
default input and output paths from :func:`tadpose.config.data_root`, so a
correctly filled ``local_paths.json`` (see :doc:`installation`) is all that
is required to run a stage on a new machine.

Using the library
------------------

The core feature functions are pure and importable.  For example,
decomposing centre-of-mass motion into body-frame velocity components:

.. code-block:: python

   import numpy as np
   from tadpose import feature_extraction as fe

   com = np.array([[0.0, 0.0], [2.0, 0.0], [4.0, 0.0]])  # (N, 2) pixels
   yaw = fe.compute_yaw(frons_xy, tail_base_xy)           # body orientation
   vel = fe.compute_velocity(com, yaw, fps=50.0, px_diameter=340.0)
   # -> {"thrust": ..., "slip": ..., "yaw_speed": ...}

Figures are written through :func:`tadpose.viz_constants.save_figure`,
which exports an editable-text SVG and a PNG (and an optional CSV data
companion) using the Wong (2011) colourblind-safe palette.

High-performance computing
--------------------------

Pose estimation and clustering at scale are designed for a SLURM cluster.
The submit scripts under ``slurm/`` source ``slurm/load_paths.sh`` so that
the interpreter, code root, data root and account come from the same
``local_paths.json`` used by the Python package, keeping every ``#SBATCH``
line machine-agnostic.
