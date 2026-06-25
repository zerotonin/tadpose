Installation
============

Environment
-----------

TadPose targets Python 3.10–3.12.  The recommended route is the bundled
conda environment, which pins the scientific stack:

.. code-block:: bash

   conda env create -f environment.yml
   conda activate tadpose
   pip install -e .

GPU-accelerated k-means (via RAPIDS cuML) is optional and installed
separately:

.. code-block:: bash

   pip install -e ".[gpu]"

For development (tests, linting, and the docs build):

.. code-block:: bash

   pip install -e ".[dev]"

Dependencies
------------

Pose estimation relies on
`DeepLabCut <https://github.com/DeepLabCut/DeepLabCut>`_, which is heavy and
installed independently of TadPose (typically in its own environment).
Statistical comparisons use
`reRandomStats <https://github.com/zerotonin/reRandomStats>`_, which is not
on PyPI and is pulled directly from GitHub by the package metadata.

Configure machine-specific paths (required first step)
------------------------------------------------------

TadPose hard-codes no absolute paths.  Every data root, interpreter, and
HPC setting is read from a gitignored ``local_paths.json``, resolved
against a committed template.  Copy it and edit the ``local`` profile (and,
on a cluster, the ``hpc`` profile) **before running anything**:

.. code-block:: bash

   cp local_paths.template.json local_paths.json
   # then edit local_paths.json: data_root, code_root, python_interpreter, …

The data root resolves in the order: ``$TADPOSE_DATA_ROOT`` →
the active profile's ``data_root`` in ``local_paths.json`` → an in-repo
``data/`` symlink.  A missing ``local_paths.json`` fails loudly and names
the template to copy.  SLURM submit scripts read the same file through
``slurm/load_paths.sh``.
